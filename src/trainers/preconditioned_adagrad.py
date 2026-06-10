from collections import defaultdict
from copy import deepcopy
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
from utils import coarsen_graph, get_data

from .common import (
    GAMMA_ALGO,
    WEIGHTING_STRATEGY,
    EarlyStopping,
    Trainer,
    apply_to_models,
    build_model,
    cycle,
)


class Preconditioned_Adagrad(Trainer):
    """Partitioned graph preconditioner, gradient accumulation variation"""

    def __init__(
        self,
        pre_epochs: int,
        full_batches: str | int,
        coarse_batches: int,
        part_trainloader: DataLoader,
        num_parts: int,
        gamma_algo: GAMMA_ALGO,
        optim_params: dict,
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
        self.pre_steps = pre_epochs  # steps over partitioned graph data
        self.full_batches = full_batches
        self.coarse_batches = coarse_batches
        self.part_trainloader = part_trainloader
        self.num_parts = num_parts
        self.gamma_algo = gamma_algo
        self.pre_lr = pre_lr
        self.pre_wd = pre_wd
        self.batched = batched
        self.ll_resolution = ll_resolution
        self.gamma_lr = gamma_lr
        self.gamma_strat = gamma_strat
        self.optim_params = optim_params

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

        optimizer = DD_Adagrad(model.parameters(), higher_state, **self.optim_params)

        model.train()

        optimizer.zero_grad()
        for data in tqdm(
            cycle(self.part_trainloader, self.pre_steps),
            desc=f"P{i} E{epoch:03}",
            dynamic_ncols=True,
            leave=False,
            disable=self.quiet,
            total=self.pre_steps,
        ):

            def closure():
                out, y = model(**get_data(data, i, self.device))
                loss = model.loss(out, y)["loss"]
                return loss

            optimizer.step(closure)
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
        self.model.to(self.device)
        optimizer = DD_Adagrad(self.model.parameters(), **self.optim_params)
        if isinstance(self.full_batches, int):
            trainloader = cycle(self.trainloader)
        else:
            trainloader = self.trainloader

        valid_loss = defaultdict(float)
        k_iter = 0
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
                ),
                start=1,
            ):
                data.to(self.device)

                def closure():
                    out, y = self.model(**get_data(data))
                    return self.model.loss(out, y)["loss"]

                loss = optimizer.step(closure)
                k_iter += 1
                train_loss += loss.detach().item()
                optimizer.zero_grad()

                if isinstance(self.full_batches, int) and i >= self.full_batches:
                    break

            train_loss /= (
                self.full_batches
                if isinstance(self.full_batches, int)
                else len(self.trainloader)
            )

            # Validation
            valid_loss = self.validate(self.model)

            if not self.quiet:
                print(f"Epoch: {epoch:03} | Valid Loss: {valid_loss['loss']}")

            mlflow.log_metrics(
                {"train/loss": train_loss}
                | {f"validate/{k}": v for k, v in valid_loss.items()},
                step=k_iter,
            )

            # mlflow.log_metrics(
            #     {f"validate/scaled_{k}": v for k, v in valid_loss.items()},
            #     step=scaled_epochs - 1,
            # )

            # Coarse step
            if self.coarse_batches > 0:
                coarse_optim = DD_Adagrad(
                    self.model.parameters(),
                    higher_state=optimizer.state_dict(),
                    **self.optim_params,
                )
                for i, data in enumerate(
                    tqdm(
                        trainloader,
                        desc=f"Coarse Epoch: {epoch:03}",
                        dynamic_ncols=True,
                        disable=self.quiet,
                        total=self.coarse_batches,
                    ),
                    start=1,
                ):
                    data.to(self.device)
                    coarse_data = coarsen_graph(data)
                    coarse_data.y = data.y

                    def closure():
                        out, y = self.model(**get_data(coarse_data))
                        return self.model.loss(out, y)["loss"]

                    loss = coarse_optim.step(closure)
                    train_loss += loss.detach().item()
                    coarse_optim.zero_grad()

                    if i >= self.coarse_batches:
                        k_iter += int(ceil(self.coarse_batches / 2))
                        break

                # update top level weights
                def closure_full_grad():
                    loss = 0
                    for data in tqdm(
                        self.validloader,
                        desc=f"Epoch: {epoch:03}",
                        dynamic_ncols=True,
                        disable=self.quiet,
                    ):
                        data.to(self.device)

                        out, y = self.model(**get_data(data))
                        loss = loss + self.model.loss(out, y)["loss"]
                    return loss

                optimizer.update_weights(closure_full_grad)

            # Preconditioning step

            contributions = [
                self.precondition(
                    model_g=self.model,
                    higher_state=optimizer.state_dict(),  # NOTE: uses the last batch gradient before update
                    i=i,
                    epoch=epoch,
                )
                for i in range(self.num_parts)
            ]

            k_iter += int(ceil(self.pre_steps / self.num_parts))

            # Contribution combination
            if self.gamma_algo == GAMMA_ALGO.SGD:
                # FIXME: this adds cost and needs clamping
                gammas = self.optimize_gammas(contributions, epoch)  # model.eval()
            else:
                gammas = [1 / self.num_parts] * self.num_parts

            mlflow.log_metrics(
                {f"gamma/p{i}": gamma for i, gamma in enumerate(gammas)},
                step=epoch,
            )

            w_avg = deepcopy(self.model.state_dict())
            w_avg = build_model(self.gamma_strat, w_avg, gammas, contributions)

            self.model.load_state_dict(w_avg)

            # scaled_epochs += int(ceil(self.pre_epochs / self.num_parts))

            # LOGGING
            vloss = self.validate(self.model)  # model.eval()
            mlflow.log_metrics(
                {f"after_pre/{k}": v for k, v in vloss.items()},
                step=epoch,
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
        gamma_optim = self.optim(params=[gammas], lr=self.gamma_lr / self.pre_steps)

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
                theta = build_model(self.gamma_strat, theta, gammas, contributions)

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
