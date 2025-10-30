from copy import deepcopy
from enum import StrEnum, auto
from typing import Any, Callable

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm
from mlflow.pytorch import log_model
from numpy import ceil
from scipy.optimize import minimize_scalar
from torch.func import functional_call
from torch.linalg import vector_norm
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.common import GNN
from utils import get_data

from .common import EarlyStopping, Trainer


class GAMMA_ALGO(StrEnum):
    """Contribution combination algorithm that determines the gamma weights"""

    NONE = auto()
    BACKTRACKING = auto()
    BRENT = auto()
    SGD = "SGD"


def apply_to_models(a: dict, fun: Callable, b: dict | None = None):
    """Apply `fun` to `a` model state dictionary (inplace)"""
    for key in a:
        if a[key].data.dtype == torch.float:
            a[key] = fun(a[key]) if b is None else fun(a[key], b[key])


def parameter_norm(params: dict) -> float:
    norm = vector_norm(
        torch.cat([p.view(-1) for p in params.values() if p is not None])
    )
    return norm.item()


def parameter_dot(grad: dict, params: dict) -> float:
    a = torch.cat([g.view(-1) for g in grad.values() if g is not None])
    b = torch.cat([params[k].view(-1) for k in grad.keys() if params[k] is not None])
    return torch.dot(a, b).item()


class Preconditioned(Trainer):
    """Partitioned graph preconditioner, gradient accumulation variation"""

    def __init__(
        self,
        pre_epochs: int,
        full_epochs: int,
        part_trainloader: DataLoader,
        num_parts: int,
        ASM: bool,
        gamma_algo: GAMMA_ALGO,
        pre_lr: float = 0,
        pre_wd: float = 0,
        batched: bool = False,
        target: str = "train",
        ll_resolution: int = 20,
        gamma_lr: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pre_epochs = pre_epochs  # epochs over partitioned graph data
        self.full_epochs = full_epochs  # epochs over full graph data
        self.part_trainloader = part_trainloader
        self.num_parts = num_parts
        self.ASM = ASM
        self.gamma_algo = gamma_algo
        self.pre_lr = pre_lr
        self.pre_wd = pre_wd
        self.batched = batched
        self.ll_resolution = ll_resolution
        self.gamma_lr = gamma_lr

        if target == "train":
            self.targetloader = self.trainloader
        else:
            self.targetloader = self.validloader

    def precondition(
        self,
        model_g: GNN,
        lr: float,
        optim_state: dict,
        i: int,
        epoch: int,
        grad: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """
        i: partition index
        epoch: global epoch for logging
        returns: difference in model weights after the preconditioning
        """
        model = deepcopy(model_g).to(self.device)
        weights_0 = deepcopy(model_g.state_dict())
        model.train()

        pre_optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=self.pre_wd
        )
        pre_optimizer.load_state_dict(optim_state)

        pre_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            pre_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        pre_optimizer.zero_grad()
        for pre_epoch in range(self.pre_epochs):
            pre_train_loss = 0

            for data in tqdm(
                self.part_trainloader,
                desc=f"P{i} E{epoch:03}: {pre_epoch:02}",
                dynamic_ncols=True,
                leave=False,
                disable=self.quiet,
            ):
                out, y = model(**get_data(data, i, self.device))
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

            # if pre_epoch % 19 == 0:
            #     acc, vloss = self.validate(model)  # model.eval
            #     mlflow.log_metrics(
            #         {
            #             f"pre_{i}/loss": vloss,
            #             **({f"pre_{i}/acc": acc} if acc is not None else {}),
            #         },
            #         step=epoch * self.pre_epochs + pre_epoch,
            #     )

            # delta_w = deepcopy(model.state_dict())
            # apply_to_models(
            #     delta_w,
            #     lambda a, b: a - b,
            #     weights_0,
            # )
            # dot = parameter_dot(grad, delta_w)
            # mlflow.log_metrics(
            #     {
            #         f"pre_train/loss_p{i}": pre_train_loss,
            #         f"pre_train/lr_p{i}": pre_scheduler.get_last_lr()[0],
            #         f"pre_{i}/dot": dot,
            #     },
            #     step=epoch * self.pre_epochs + pre_epoch,
            #     # step=epoch * (self.pre_epochs + 1) + pre_epoch,
            # )

        # computing the weight difference
        delta_w = deepcopy(model.state_dict())
        apply_to_models(
            delta_w,
            lambda a, b: a - b,
            weights_0,
        )
        return delta_w

    def optimal_combination(
        self, model: GNN, contribution: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """
        Combines model weights with a contribution in the most optimal way.
        Uses line search algorithms to find the weight gamma with which the contribution will be added.
        """
        weights = deepcopy(model.state_dict())
        model = deepcopy(model).to(self.device)

        # HACK
        def objective(gamma: float):
            w_new = deepcopy(weights)
            apply_to_models(
                w_new,
                lambda a, b: a + gamma * b,
                contribution,
            )
            model.load_state_dict(w_new)
            _, loss = self.validate(model, self.targetloader)
            return loss

        gamma = 1.0
        match self.gamma_algo:
            case GAMMA_ALGO.BRENT:
                gamma = minimize_scalar(
                    objective,
                    bounds=(0, 1),
                    method="bounded",
                    options={
                        "maxiter": 10,
                        "xatol": 1e-3,
                    },
                ).x  # type: ignore

            case GAMMA_ALGO.BACKTRACKING:
                beta = 0.5
                _, fx = self.validate(model, self.targetloader)
                for _ in range(10):
                    if objective(gamma) < fx:
                        break
                    gamma *= beta
                else:  # no valid gamma found, disable contribution
                    return weights, 0.0

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
        scaled_epochs = 0
        for epoch in range(self.epochs):
            grad_train = self.get_global_grad(epoch, self.trainloader)
            grad_norm = parameter_norm(grad_train)

            # LOGGING
            acc, vloss = self.validate(self.model)
            mlflow.log_metrics(
                {
                    "before_pre/loss": vloss,
                    **({"before_pre/acc": acc} if acc is not None else {}),
                    "grad/global_L2": grad_norm,
                },
                step=epoch,
            )

            w0 = deepcopy(self.model.state_dict())

            # Preconditioning step
            if self.ASM:  # Additive Schwarz
                contributions = []

                for i in range(self.num_parts):
                    delta_w = self.precondition(
                        model_g=self.model,
                        # self.pre_lr,
                        lr=scheduler.get_last_lr()[0],  # pass down the lr
                        optim_state=deepcopy(optimizer.state_dict()),
                        i=i,
                        epoch=epoch,
                        grad=grad_train,
                    )
                    contributions.append(delta_w)

                    dot = parameter_dot(grad_train, delta_w)
                    mlflow.log_metrics(
                        {
                            f"grad/p{i}_dot": dot,
                        },
                        step=epoch,
                    )

                # Contribution combination
                if self.gamma_algo == GAMMA_ALGO.SGD:
                    gammas = self.optimize_gammas(contributions, epoch)  # model.eval()

                    for i, gamma in enumerate(gammas):
                        mlflow.log_metrics(
                            {
                                f"gamma/p{i}": gamma,
                            },
                            step=epoch,
                        )

                    w_avg = deepcopy(self.model.state_dict())
                    for delta_w, gamma in zip(contributions, gammas):
                        apply_to_models(
                            w_avg,
                            lambda a, b: a + gamma * b,
                            delta_w,
                        )
                    self.model.load_state_dict(w_avg)

                # NOTE: combines contributions one at a time
                else:  # Regular line search
                    w_avg = deepcopy(self.model.state_dict())
                    for i, delta_w in enumerate(contributions):
                        if self.gamma_algo == GAMMA_ALGO.NONE:
                            gamma = 1 / len(contributions)  # equal weights for all
                            apply_to_models(
                                w_avg,
                                lambda a, b: a + gamma * b,
                                delta_w,
                            )
                        else:
                            w_new, gamma = self.optimal_combination(self.model, delta_w)

                            self.model.load_state_dict(w_new)
                            mlflow.log_metrics(
                                {
                                    f"gamma/p{i}": gamma,
                                },
                                step=epoch,
                            )

                    if self.gamma_algo == GAMMA_ALGO.NONE:
                        self.model.load_state_dict(w_avg)

            else:  # Multiplicative Schwarz
                for i in range(self.num_parts):
                    if i != 0:
                        grad_train = self.get_global_grad(epoch, self.trainloader)

                    delta_w = self.precondition(
                        model_g=self.model,
                        lr=scheduler.get_last_lr()[0],  # pass down the lr
                        optim_state=deepcopy(optimizer.state_dict()),
                        i=i,
                        epoch=epoch,
                        grad=grad_train,
                    )
                    w_new, gamma = self.optimal_combination(self.model, delta_w)

                    self.model.load_state_dict(w_new)
                    dot = parameter_dot(grad_train, delta_w)
                    mlflow.log_metrics(
                        {
                            f"gamma/p{i}": gamma,
                            f"grad/p{i}_dot": dot,
                        },
                        step=epoch,
                    )
                    # self.train(optimizer, epoch)  # TEST: 05-22 notes

            # in case of Additive Schwarz the parts can be ran in parallel
            if self.ASM:
                scaled_epochs += int(ceil(self.pre_epochs / self.num_parts))
            else:
                scaled_epochs += self.pre_epochs

            diff = deepcopy(self.model.state_dict())
            apply_to_models(
                diff,
                lambda a, b: a - b,
                w0,
            )
            pre_norm = parameter_norm(diff)
            dot = parameter_dot(grad_train, diff)
            mlflow.log_metrics(
                {
                    "grad/pre_dot": dot,
                    "grad/pre_L2": pre_norm,
                },
                step=epoch,
            )

            # LOGGING
            acc, vloss = self.validate(self.model)  # model.eval()
            mlflow.log_metrics(
                {
                    "after_pre/loss": vloss,
                    **({"after_pre/acc": acc} if acc is not None else {}),
                },
                step=epoch,
            )

            # Full pass
            train_loss = 0
            for _ in range(self.full_epochs):
                train_loss = self.train(optimizer, epoch)  # model.train()
                mlflow.log_metrics(
                    {
                        "train/scaled_loss": train_loss,
                    },
                    step=scaled_epochs,
                )
                scaled_epochs += 1

            # Validation
            acc, valid_loss = self.validate(self.model)

            scheduler.step(valid_loss)

            if not self.quiet:
                print(f"Epoch: {epoch:03} | Valid Loss: {valid_loss}")

            mlflow.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "validate/loss": valid_loss,
                    **({"validate/accuracy": acc} if acc is not None else {}),
                },
                step=epoch,
            )

            mlflow.log_metrics(
                {
                    "validate/scaled_loss": valid_loss,
                    **({"validate/scaled_accuracy": acc} if acc is not None else {}),
                },
                step=scaled_epochs - 1,
            )

            if epoch % 10 == 9:
                log_model(self.model, "model")

        accuracy = self.test()
        if not self.quiet:
            print(f"Accuracy: {accuracy}")

        mlflow.log_metric("test/accuracy", accuracy)

        return valid_loss

    def get_global_grad(self, epoch, dataloader) -> dict[str, torch.Tensor]:
        """Computes a global gradient"""
        self.model.train()
        self.model.zero_grad()
        for data in tqdm(
            dataloader,
            desc=f"Gradient: {epoch:03}",
            dynamic_ncols=True,
            leave=False,
            disable=self.quiet,
        ):
            data.to(self.device)

            out, y = self.model(**get_data(data))
            loss = self.model.loss(out, y)

            loss.backward()

        return {
            name: deepcopy(param.grad)
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

    def train(self, optimizer, epoch):
        """Global training step"""
        train_loss = 0
        self.model.train()

        optimizer.zero_grad()
        for data in tqdm(
            self.trainloader,
            desc=f"Epoch: {epoch:03}",
            dynamic_ncols=True,
            disable=self.quiet,
        ):
            data.to(self.device)

            out, y = self.model(**get_data(data))
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

        train_loss /= len(self.trainloader)
        return train_loss

    # NOTE: Causes sharp deterioration in performance when stuck in local minimum (starting with ones)
    # BUG: exploding gradient
    def optimize_gammas(self, contributions, global_epoch):
        """Does a mini optimization to find the best gamma combination"""

        if self.num_parts == 2 and (
            global_epoch in [1, 2, 3, 4] or global_epoch % 10 == 0
        ):
            self.loss_landscape(contributions, global_epoch, grid_n=self.ll_resolution)

        N_EPOCHS = 1000
        # gammas = torch.ones(self.num_parts, requires_grad=True)
        gammas = torch.zeros(
            self.num_parts, requires_grad=True
        )  # start with no contributions
        gamma_history = [gammas.detach().cpu().clone().numpy()]
        gamma_optim = self.optim(params=[gammas], lr=self.gamma_lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            gamma_optim,
            mode="min",
            factor=0.5,
            patience=5,
        )

        es = EarlyStopping(patience=10)

        self.model.eval()
        params = {}
        buffers = {}
        for key, val in self.model.state_dict().items():
            if key in dict(self.model.named_buffers()):
                buffers[key] = val.clone()
            else:
                params[key] = val.clone()

        for epoch in range(N_EPOCHS):
            gamma_optim.zero_grad()

            valid_loss = 0
            # correct = 0
            # total = 0

            for data in tqdm(
                self.targetloader,
                desc=f"Eval {epoch}",
                dynamic_ncols=True,
                leave=False,
                disable=self.quiet,
            ):
                data.to(self.device)
                # Gamma.T * Theta
                # NOTE: rebuilding required for computational graph
                theta = deepcopy(params)
                for i, delta_w in enumerate(contributions):
                    apply_to_models(
                        theta,
                        lambda a, b: a + gammas[i] * b,
                        delta_w,
                    )

                out, y = functional_call(
                    self.model, (theta, buffers), kwargs=get_data(data)
                )
                loss = self.model.loss(out, y)
                valid_loss += loss.detach().item()

                (grad,) = torch.autograd.grad(loss, gammas)
                gammas.grad = grad.detach()
                gamma_optim.step()

                gamma_history.append(gammas.detach().cpu().clone().numpy())

                # # accuracy
                # pred = out.argmax(dim=1)  # Predicted labels
                # correct += (pred == y).sum().item()
                # total += y.size(0)

            valid_loss /= len(self.targetloader)
            mlflow.log_metrics(
                {
                    "gamma_optim/loss": valid_loss,
                    # "gamma_optim/accuracy": correct / total,
                },
                # NOTE: early stopping causes gaps in the plot
                step=epoch + N_EPOCHS * global_epoch,
            )

            scheduler.step(valid_loss)
            if es.step(valid_loss):
                break

        path = f"results/{mlflow.active_run().info.run_id}_gamma_history_{global_epoch:03}.csv"  # type: ignore
        df = pd.DataFrame(gamma_history)
        df.to_csv(path)
        mlflow.log_artifact(path)

        return gammas.detach().cpu().numpy()

    def loss_landscape(
        self,
        contributions,
        global_epoch,
        grid_n: int,
        gamma_min: float = -0.5,
        gamma_max: float = 1.5,
    ):
        if grid_n <= 0:
            return

        self.model.eval()
        params = {}
        buffers = {}
        for key, val in self.model.state_dict().items():
            if key in dict(self.model.named_buffers()):
                buffers[key] = val.clone()
            else:
                params[key] = val.clone()

        def loss_fn(gammas):
            theta = deepcopy(params)
            for i, delta_w in enumerate(contributions):
                apply_to_models(
                    theta,
                    lambda a, b: a + gammas[i] * b,
                    delta_w,
                )

            total_loss = torch.tensor([0.0], device=self.device)
            for data in tqdm(
                self.targetloader,
                dynamic_ncols=True,
                leave=False,
                disable=self.quiet,
            ):
                data.to(self.device)

                out, y = functional_call(
                    self.model, (theta, buffers), kwargs=get_data(data)
                )
                loss = self.model.loss(out, y)
                total_loss += loss
            return total_loss / len(self.targetloader)

        X = np.linspace(gamma_min, gamma_max, grid_n)
        Z = np.ndarray((grid_n, grid_n))
        for idx in range(grid_n):
            for idy in range(grid_n):
                Z[idx, idy] = loss_fn(torch.tensor([X[idx], X[idy]])).detach().item()

        # FIXME: average out the batched version
        # # Compute Hessian
        # gammas = torch.zeros(self.num_parts, requires_grad=True)
        # H = torch.autograd.functional.hessian(loss_fn, gammas)  # type: ignore
        # H = H.detach().cpu().numpy()  # type: ignore
        # mlflow.log_table(
        #     pd.DataFrame(H), artifact_file=f"hessian_{global_epoch:03}.json"
        # )

        # Plot and log the loss landscape
        fig, ax = plt.subplots()
        im = ax.imshow(
            Z,
            extent=(gamma_min, gamma_max, gamma_min, gamma_max),
            cmap="managua",
            norm=LogNorm(vmin=Z.min() + 1e-10, vmax=Z.max()),
        )
        fig.colorbar(im, label="Loss")

        fig.tight_layout()
        mlflow.log_figure(
            fig,
            artifact_file=f"loss_landscape_{global_epoch:03}.pdf",
        )

        path = f"results/{mlflow.active_run().info.run_id}_losses_{global_epoch:03}.csv"  # type: ignore
        df = pd.DataFrame(Z, columns=X)
        df.to_csv(path)
        mlflow.log_artifact(path)
