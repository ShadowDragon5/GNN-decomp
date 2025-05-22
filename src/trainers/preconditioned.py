from copy import deepcopy
from enum import Enum
from typing import Any, Callable

import mlflow
import torch
from scipy.optimize import minimize_scalar
from torch.nn import Module
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import PartitionedData

from .common import Trainer


class LS_ALGO(Enum):
    """Line Search algorithm"""

    BACKTRACKING = "backtracking"
    BRENT = "brent"


def apply_to_models(a: dict, fun: Callable, b: dict | None = None):
    """Apply `fun` to `a` model state dictionary (inplace)"""
    for key in a:
        if a[key].data.dtype == torch.float:
            a[key] = fun(a[key]) if b is None else fun(a[key], b[key])


class Preconditioned(Trainer):
    """Partitioned graph preconditioner, gradient accumulation variation"""

    def __init__(
        self,
        pre_epochs: int,
        part_trainloader: DataLoader,
        num_parts: int,
        ASM: bool,
        pre_lr: float = 0,
        pre_wd: float = 0,
        batched: bool = False,
        ls_algo: LS_ALGO = LS_ALGO.BACKTRACKING,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pre_epochs = pre_epochs
        self.num_parts = num_parts
        self.part_trainloader = part_trainloader
        self.pre_lr = pre_lr
        self.pre_wd = pre_wd
        self.ASM = ASM
        self.batched = batched
        self.ls_algo = ls_algo

    def precondition(
        self, model: Module, lr: float, i: int, epoch: int
    ) -> dict[str, Any]:
        """
        returns: difference in model weights after the preconditioning
        """
        model = deepcopy(model).to(self.device)
        weights_0 = deepcopy(model.state_dict())

        pre_optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=self.pre_wd
        )

        pre_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            pre_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        for pre_epoch in range(self.pre_epochs):
            pre_train_loss = 0

            for data in tqdm(
                self.part_trainloader,
                desc=f"P{i} E{epoch:03}: {pre_epoch:02}",
                dynamic_ncols=True,
                leave=False,
                disable=self.quiet,
            ):
                data: PartitionedData = data
                x = data.get_x(i, self.device).to(self.device)
                edge_index = data.get_edge_index(i, self.device)
                batch = data.get_batch(i, self.device)
                y = data.get_y(i, self.device)

                out = model(x, edge_index, batch)
                loss = model.loss(out, y)

                pre_train_loss += loss.detach().item()

                if not self.batched:
                    loss = loss / len(self.part_trainloader)

                loss.backward()
                if self.batched:
                    pre_optimizer.step()
                    pre_optimizer.zero_grad()

            if not self.batched:
                pre_optimizer.step()
                pre_optimizer.zero_grad()

            pre_train_loss /= len(self.part_trainloader)
            pre_scheduler.step(pre_train_loss)

            if pre_epoch % 19 == 0:
                acc, vloss = self.validate(model)
                mlflow.log_metrics(
                    {
                        f"pre_{i}/loss": vloss,
                        f"pre_{i}/acc": acc,
                    },
                    step=epoch * self.pre_epochs + pre_epoch,
                )

            mlflow.log_metrics(
                {
                    f"pre_train/loss_p{i}": pre_train_loss,
                    f"pre_train/lr_p{i}": pre_scheduler.get_last_lr()[0],
                },
                step=epoch * self.pre_epochs + pre_epoch,
            )

        # computing the weight difference
        delta_w = deepcopy(model.state_dict())
        apply_to_models(
            delta_w,
            lambda a, b: a - b,
            weights_0,
        )
        return delta_w

    def optimal_combination(
        self, model: torch.nn.Module, contribution: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """
        Combines model weights with contribution in the most optimal way.
        Uses line search algorithms to find the weight gamma with which the contribution will be added.
        """
        weights = deepcopy(model.state_dict())
        model = deepcopy(model).to(model.device)

        # HACK
        def objective(gamma: float):
            w_new = deepcopy(weights)
            apply_to_models(
                w_new,
                lambda a, b: a + gamma * b,
                contribution,
            )
            model.load_state_dict(w_new)
            _, loss = self.validate(model)
            return loss

        gamma = 1.0
        if self.ls_algo == LS_ALGO.BRENT:
            gamma = minimize_scalar(
                objective,
                bounds=(0, 1),
                method="bounded",
                options={
                    "maxiter": 10,
                    "xatol": 1e-3,
                },
            ).x  # type: ignore

        elif self.ls_algo == LS_ALGO.BACKTRACKING:
            beta = 0.5
            _, fx = self.validate(model)
            for _ in range(10):
                if objective(gamma) < fx:
                    break
                gamma *= beta

        apply_to_models(
            weights,
            lambda a, b: a + gamma * b,
            contribution,
        )
        return weights, gamma

    def run(self) -> float:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.model.to(self.device)

        valid_loss = 0
        for epoch in range(self.epochs):
            train_loss = 0
            self.model.train()

            # LOGGING
            acc, vloss = self.validate(self.model)
            mlflow.log_metrics(
                {
                    "before_pre/loss": vloss,
                    "before_pre/acc": acc,
                },
                step=epoch,
            )

            # Preconditioning step
            if self.ASM:  # Additive Schwarz
                # FIXME: update
                models = []

                for i in range(self.num_parts):
                    delta_w = self.precondition(
                        self.model,
                        # self.pre_lr,
                        scheduler.get_last_lr()[0],  # pass down the lr
                        i,
                        epoch,
                    )
                    models.append(delta_w)

                # w_avg = deepcopy(w_0)
                for i, delta_w in enumerate(models):
                    # gamma = 1 / len(models)
                    # apply_to_models(
                    #     w_avg,
                    #     lambda a, b: a + gamma * b,
                    #     delta_w,
                    # )
                    # NOTE: combines contributions one at a time
                    w_new, gamma = self.optimal_combination(self.model, delta_w)

                    self.model.load_state_dict(w_new)
                    mlflow.log_metrics(
                        {
                            f"gamma/p{i}": gamma,
                        },
                        step=epoch,
                    )

                # self.model.load_state_dict(w_avg)

            else:  # Multiplicative Schwarz
                for i in range(self.num_parts):
                    delta_w = self.precondition(
                        self.model,
                        # self.pre_lr,
                        scheduler.get_last_lr()[0],  # pass down the lr
                        i,
                        epoch,
                    )
                    w_new, gamma = self.optimal_combination(self.model, delta_w)

                    self.model.load_state_dict(w_new)
                    mlflow.log_metrics(
                        {
                            f"gamma/p{i}": gamma,
                        },
                        step=epoch,
                    )

            # LOGGING
            acc, vloss = self.validate(self.model)
            mlflow.log_metrics(
                {
                    "after_pre/loss": vloss,
                    "after_pre/acc": acc,
                },
                step=epoch,
            )

            # Full pass
            for data in tqdm(
                self.trainloader,
                desc=f"Epoch: {epoch:03}",
                dynamic_ncols=True,
                disable=self.quiet,
            ):
                x = data.x.to(self.device)
                y = data.y.to(self.device)

                edge_index = data.edge_index.to(self.device)
                batch = data.batch.to(self.device)

                out = self.model(x, edge_index, batch)
                loss = self.model.loss(out, y)

                train_loss += loss.detach().item()

                if not self.batched:
                    loss = loss / len(self.trainloader)

                loss.backward()
                if self.batched:
                    optimizer.step()
                    optimizer.zero_grad()

            if not self.batched:
                optimizer.step()
                optimizer.zero_grad()

            train_loss /= len(self.trainloader)

            # Validation
            accuracy, valid_loss = self.validate(self.model)

            scheduler.step(valid_loss)

            if not self.quiet:
                print(f"Epoch: {epoch:03} | " f"Valid Loss: {valid_loss}")

            mlflow.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "validate/loss": valid_loss,
                    "validate/accuracy": accuracy,
                },
                step=epoch,
            )

        accuracy = self.test()
        if not self.quiet:
            print(f"Accuracy: {accuracy}")

        mlflow.log_metric("test/accuracy", accuracy)

        return valid_loss
