from collections import defaultdict
from copy import deepcopy
from itertools import cycle
from typing import Any

import mlflow
import pandas as pd
import torch
from numpy import ceil
from torch.func import functional_call
from torch.optim.optimizer import StateDict
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.common import GNN
from optimizers.dd_adagrad import DD_Adagrad
from utils import get_data

from .common import (
    GAMMA_ALGO,
    WEIGHTING_STRATEGY,
    EarlyStopping,
    Trainer,
    apply_to_models,
)


class Preconditioned_Adagrad(Trainer):
    """Partitioned graph preconditioner, gradient accumulation variation"""

    def __init__(
        self,
        pre_epochs: int,
        full_epochs: int,
        full_batches: str | int,
        part_trainloader: DataLoader,
        num_parts: int,
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
        self.full_batches = full_batches
        self.part_trainloader = part_trainloader
        self.num_parts = num_parts
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
        higher_state: StateDict,
        i: int,
        epoch: int,
    ) -> dict[str, Any]:
        """
        i: partition index
        epoch: global epoch for logging
        returns: difference in model weights after the preconditioning
        """
        model = deepcopy(model_g).to(self.device)
        weights_0 = deepcopy(model_g.state_dict())

        optimizer = DD_Adagrad(model.parameters(), higher_state)

        model.train()

        optimizer.zero_grad()
        for pre_epoch in range(self.pre_epochs):
            pre_train_loss = 0

            for data in tqdm(
                self.part_trainloader,
                desc=f"P{i} E{epoch:03}: {pre_epoch:02}",
                dynamic_ncols=True,
                leave=False,
                disable=self.quiet,
            ):

                def closure():
                    out, y = model(**get_data(data, i, self.device))
                    loss = model.loss(out, y)["loss"]
                    return loss

                loss = optimizer.step(closure)
                pre_train_loss += loss.detach().item()

                optimizer.zero_grad()

        # computing the weight difference
        delta_w = deepcopy(model.state_dict())
        apply_to_models(
            delta_w,
            lambda a, b: a - b,
            weights_0,
        )
        return delta_w

    def run(self) -> float:
        optimizer = DD_Adagrad(self.model.parameters())
        if isinstance(self.full_batches, int):
            trainloader = cycle(self.trainloader)
        else:
            trainloader = self.trainloader

        self.model.to(self.device)

        valid_loss = defaultdict(float)
        scaled_epochs = 0
        for epoch in range(self.epochs):
            # Taylor step
            train_loss = 0
            self.model.train()

            optimizer.zero_grad()

            for i, data in enumerate(
                tqdm(
                    trainloader,
                    desc=f"Epoch: {epoch:03}",
                    dynamic_ncols=True,
                    disable=self.quiet,
                    total=self.full_batches
                    if isinstance(self.full_batches, int)
                    else None,
                )
            ):
                data.to(self.device)

                def closure():
                    out, y = self.model(**get_data(data))
                    return self.model.loss(out, y)["loss"]

                loss = optimizer.step(closure)
                train_loss += loss.detach().item()
                optimizer.zero_grad()

                if isinstance(self.full_batches, int) and i >= self.full_batches:
                    break

            train_loss /= len(self.trainloader)

            # Preconditioning step

            # optimizer.zero_grad()
            # # Compute gradient for inner optimizer initialization
            # self.model.eval()
            # for data in tqdm(
            #     self.trainloader,
            #     desc=f"Epoch: {epoch:03}",
            #     dynamic_ncols=True,
            #     disable=self.quiet,
            # ):
            #     data.to(self.device)
            #     out, y = self.model(**get_data(data))
            #     loss = self.model.loss(out, y)["loss"]
            #     loss.backward()
            #     break
            # theta1, theta2 = optimizer.get_thetas(self.model.parameters())
            # optimizer.zero_grad()

            contributions = [
                self.precondition(
                    model_g=self.model,
                    higher_state=optimizer.state_dict(),  # NOTE: uses the last batch gradient before update
                    i=i,
                    epoch=epoch,
                )
                for i in range(self.num_parts)
            ]

            # Contribution combination
            if self.gamma_algo == GAMMA_ALGO.SGD:
                gammas = self.optimize_gammas(contributions, epoch)  # model.eval()
            else:
                gammas = [1 / self.num_parts] * self.num_parts

            mlflow.log_metrics(
                {f"gamma/p{i}": gamma for i, gamma in enumerate(gammas)},
                step=epoch,
            )

            w_avg = deepcopy(self.model.state_dict())
            w_avg = self.build_model(w_avg, gammas, contributions)

            self.model.load_state_dict(w_avg)

            scaled_epochs += int(ceil(self.pre_epochs / self.num_parts))

            # # LOGGING
            # vloss = self.validate(self.model)  # model.eval()
            # mlflow.log_metrics(
            #     {f"after_pre/{k}": v for k, v in vloss.items()},
            #     step=epoch,
            # )

            mlflow.log_metrics(
                {
                    "train/scaled_loss": train_loss,
                },
                step=scaled_epochs,
            )
            scaled_epochs += 1

            # Validation
            valid_loss = self.validate(self.model)

            if not self.quiet:
                print(f"Epoch: {epoch:03} | Valid Loss: {valid_loss['loss']}")

            mlflow.log_metrics(
                {"train/loss": train_loss}
                | {f"validate/{k}": v for k, v in valid_loss.items()},
                step=epoch,
            )

            mlflow.log_metrics(
                {f"validate/scaled_{k}": v for k, v in valid_loss.items()},
                step=scaled_epochs - 1,
            )

        return valid_loss["loss"]

    def optimize_gammas(self, contributions, global_epoch):
        """Does a mini optimization to find the best gamma combination"""

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

            valid_loss /= len(self.targetloader)
            mlflow.log_metrics(
                {
                    "gamma_optim/loss": valid_loss,
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
