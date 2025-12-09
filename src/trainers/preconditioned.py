from collections import defaultdict
from copy import deepcopy
from enum import StrEnum, auto
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd
import torch
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


class WEIGHTING_STRATEGY(StrEnum):
    """Gamma function used for combining weighting the contributions"""

    DIRECT = auto()
    CLIPPED = auto()
    INVERSE = auto()


def apply_to_models(a: dict, fun: Callable, b: dict | None = None, indexed=False):
    """Apply `fun` to `a` model state dictionary (inplace)"""
    for l, key in enumerate(reversed(a)):  # L -> 1
        if a[key].data.dtype == torch.float:
            if b is None:
                a[key] = fun(a[key])
            elif indexed:
                a[key] = fun(a[key], b[key], l)
            else:
                a[key] = fun(a[key], b[key])


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
        gamma_strat: WEIGHTING_STRATEGY = WEIGHTING_STRATEGY.DIRECT,
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
        self.gamma_strat = gamma_strat

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

        if self.batched:
            nsteps = len(self.part_trainloader) * self.pre_epochs
        else:
            nsteps = self.pre_epochs
        pre_scheduler = self.scheduler(pre_optimizer, lr, nsteps)

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
                loss = model.loss(out, y)["loss"]

                pre_train_loss += loss.detach().item()

                if not self.batched:
                    loss = loss / len(self.part_trainloader)

                loss.backward()
                if self.batched:
                    pre_optimizer.step()
                    if isinstance(pre_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        pre_scheduler.step()
                    pre_optimizer.zero_grad()

            if not self.batched:
                pre_optimizer.step()
                if isinstance(pre_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    pre_scheduler.step()
                pre_optimizer.zero_grad()

            if isinstance(pre_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                pre_train_loss /= len(self.part_trainloader)
                pre_scheduler.step(pre_train_loss)  # type: ignore

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
        def objective(gamma: float) -> float:
            w_new = deepcopy(weights)
            apply_to_models(
                w_new,
                lambda a, b: a + gamma * b,
                contribution,
            )
            model.load_state_dict(w_new)
            loss = self.validate(model, self.targetloader)["loss"]
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
                fx = self.validate(model, self.targetloader)["loss"]
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

        if self.batched:
            nsteps = len(self.trainloader) * self.epochs
        else:
            nsteps = self.epochs
        scheduler = self.scheduler(optimizer, self.lr, nsteps)

        self.model.to(self.device)

        valid_loss = defaultdict(float)
        scaled_epochs = 0
        for epoch in range(self.epochs):
            grad_train = self.get_global_grad(epoch, self.trainloader)
            grad_norm = parameter_norm(grad_train)

            # LOGGING
            vloss = self.validate(self.model)
            try:
                mlflow.log_metrics(
                    {f"before_pre/{k}": v for k, v in vloss.items()}
                    | {"grad/global_L2": grad_norm},
                    step=epoch,
                )
            except Exception:
                print("vloss:", vloss)
                print("grad_norm:", grad_norm)

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
                    w_avg = self.build_model(w_avg, gammas, contributions)

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
            vloss = self.validate(self.model)  # model.eval()
            mlflow.log_metrics(
                {f"after_pre/{k}": v for k, v in vloss.items()},
                step=epoch,
            )

            # Full pass
            train_loss = 0
            for _ in range(self.full_epochs):
                train_loss = self.train(optimizer, scheduler, epoch)  # model.train()
                mlflow.log_metrics(
                    {
                        "train/scaled_loss": train_loss,
                    },
                    step=scaled_epochs,
                )
                scaled_epochs += 1

            # Validation
            valid_loss = self.validate(self.model)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss["loss"])

            if not self.quiet:
                print(f"Epoch: {epoch:03} | Valid Loss: {valid_loss['loss']}")

            mlflow.log_metrics(
                {"train/loss": train_loss, "train/lr": scheduler.get_last_lr()[0]}
                | {f"validate/{k}": v for k, v in valid_loss.items()},
                step=epoch,
            )

            mlflow.log_metrics(
                {f"validate/scaled_{k}": v for k, v in valid_loss.items()},
                step=scaled_epochs - 1,
            )

            if epoch % 10 == 9:
                log_model(self.model, "model")

        if self.need_acc:
            accuracy = self.test()
            if not self.quiet:
                print(f"Accuracy: {accuracy}")

            mlflow.log_metric("test/accuracy", accuracy)

        return valid_loss["loss"]

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
            loss = self.model.loss(out, y)["loss"]

            loss.backward()

        return {
            name: deepcopy(param.grad)
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

    def train(self, optimizer, scheduler, epoch):
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
            loss = self.model.loss(out, y)["loss"]

            train_loss += loss.detach().item()

            if not self.batched:
                loss = loss / len(self.trainloader)

            loss.backward()
            if self.batched:
                optimizer.step()
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                optimizer.zero_grad()

        if not self.batched:
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        train_loss /= len(self.trainloader)
        return train_loss

    def optimize_gammas(self, contributions, global_epoch):
        """Does a mini optimization to find the best gamma combination"""

        if self.num_parts in [2, 3] and global_epoch in [0, 1, 2, 3, 4]:
            self.loss_landscape(contributions, global_epoch, grid_n=self.ll_resolution)

        N_EPOCHS = 1000
        EPS = 1e-8  # to prevent INVERSE inf derivative at 0
        gammas = torch.full([self.num_parts], EPS, requires_grad=True)
        # gammas = torch.ones(self.num_parts, requires_grad=True)

        # gammas = torch.cat([torch.ones(1), torch.zeros(self.num_parts - 1)])
        # gammas = gammas.requires_grad_()

        gamma_history = [gammas.detach().cpu().clone().numpy()]
        gamma_optim = self.optim(params=[gammas], lr=self.gamma_lr / self.pre_epochs)

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
                desc=f"Gamma optim. {epoch}",
                dynamic_ncols=True,
                leave=False,
                disable=self.quiet,
            ):
                data.to(self.device)
                # NOTE: rebuilding required for computational graph
                theta = deepcopy(params)
                theta = self.build_model(theta, gammas, contributions)

                out, y = functional_call(
                    self.model, (theta, buffers), kwargs=get_data(data)
                )
                loss = self.model.loss(out, y)["loss"]
                valid_loss += loss.detach().item()

                (grad,) = torch.autograd.grad(loss, gammas)
                gammas.grad = grad.detach()
                gamma_optim.step()
                gammas = torch.clamp(gammas, min=EPS)

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
        gamma_min: float = 0.0,
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
            theta = self.build_model(theta, gammas, contributions)

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
                loss = self.model.loss(out, y)["loss"]
                total_loss += loss
            return total_loss / len(self.targetloader)

        X = np.linspace(gamma_min, gamma_max, grid_n)

        if self.num_parts == 2:
            Z = np.ndarray((grid_n, grid_n))
            for idx in range(grid_n):
                for idy in range(grid_n):
                    Z[idx, idy] = (
                        loss_fn(torch.tensor([X[idx], X[idy]])).detach().item()
                    )

        elif self.num_parts == 3:
            Z = np.ndarray((grid_n, grid_n, grid_n))
            for idx in range(grid_n):
                for idy in range(grid_n):
                    for idz in range(grid_n):
                        Z[idx, idy, idz] = (
                            loss_fn(torch.tensor([X[idx], X[idy], X[idz]]))
                            .detach()
                            .item()
                        )
        else:
            return

        path = f"results/{mlflow.active_run().info.run_id}_losses_{global_epoch:03}.npy"  # type: ignore
        np.save(path, Z)
        mlflow.log_artifact(path)

    def build_model(self, theta, gammas, contributions):
        def weigthing_strategy(a, b, l, i):
            match self.gamma_strat:
                case WEIGHTING_STRATEGY.DIRECT:
                    return a + gammas[i] * b
                case WEIGHTING_STRATEGY.CLIPPED:
                    if l >= 4 * 2:  # weight + bias
                        return (a + gammas[i] * b).detach()
                    return a + gammas[i] * b
                case WEIGHTING_STRATEGY.INVERSE:
                    base = 2
                    # NOTE: gammas must be positive (>0)
                    return a + (gammas[i] ** (base**-l)) * b

        for i, delta_w in enumerate(contributions):
            apply_to_models(
                a=theta,
                fun=lambda a, b, l: weigthing_strategy(a, b, l, i),
                b=delta_w,
                indexed=True,
            )

        return theta
