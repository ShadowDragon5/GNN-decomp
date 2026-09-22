from collections import defaultdict
from copy import deepcopy
from logging import warning
from typing import Any, Callable

import mlflow
import torch
import torch.distributed as dist
from numpy import ceil
from torch.optim.optimizer import StateDict
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.common import GNN
from optimizers.dd_adagrad import DD_Adagrad
from utils import get_data

from .common import (
    GAMMA_ALGO,
    WEIGHTING_STRATEGY,
    Trainer,
    apply_to_models,
    cycle,
)


class Dist_Adagrad(Trainer):
    """Distributed preconditioned Adagrad Trainer"""

    def __init__(
        self,
        pre_epochs: int,
        full_batches: str | int,
        coarse_batches: int,
        coarse_level: int,
        coarse_fn: Callable,
        use_coarse_moment: bool,
        after_steps: int,
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
        self.coarse_level = coarse_level
        self.coarse_fn = coarse_fn
        self.use_coarse_moment = use_coarse_moment
        self.after_steps = after_steps
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

        if self.device.type == "cuda":
            dist.init_process_group("nccl")
        else:
            dist.init_process_group("gloo")

        self.world_size = dist.get_world_size()
        if self.world_size < self.num_parts:
            warning(
                "Not enough processors found."
                f"Reducing partition count to {self.world_size}"
            )
            self.num_parts = self.world_size
        elif self.world_size > self.num_parts:
            warning("Allocated more processors than partitions.")

        self.rank = dist.get_rank()

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
        model = deepcopy(model_g).to(f"{self.device}:{i % self.world_size}")
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
            full_batches = self.full_batches
        else:
            full_batches = len(self.trainloader)

        trainloader = cycle(self.trainloader)

        valid_loss = defaultdict(float)
        k_iter = 0
        for epoch in range(self.epochs):
            # Taylor step
            if self.rank == 0:
                train_loss = 0
                self.model.train()

                optimizer.zero_grad()

                # skip full graph
                if full_batches == 0:
                    # update top level weights
                    def grad_closure():
                        data = next(iter(self.trainloader))
                        data.to(self.device)
                        out, y = self.model(**get_data(data))
                        return self.model.loss(out, y)["loss"]

                    optimizer.update_weights(grad_closure)

                else:
                    for i, data in enumerate(
                        tqdm(
                            trainloader,
                            desc=f"Epoch: {epoch:03}",
                            dynamic_ncols=True,
                            disable=self.quiet,
                            total=full_batches,
                        ),
                        start=1,
                    ):
                        data.to(self.device)

                        def closure():
                            out, y = self.model(**get_data(data))
                            return self.model.loss(out, y)["loss"]

                        loss = optimizer.step(closure)
                        k_iter += 1
                        train_loss += loss.detach().item() / full_batches
                        optimizer.zero_grad()

                        if i >= full_batches:
                            break

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
                        coarse_data = self.coarse_fn(data, level=self.coarse_level)
                        coarse_data.to(self.device)  # type: ignore

                        def closure():
                            out, y = self.model(**get_data(coarse_data))
                            return self.model.loss(out, y)["loss"]

                        loss = coarse_optim.step(closure)
                        train_loss += loss.detach().item()
                        coarse_optim.zero_grad()

                        if i >= self.coarse_batches:
                            k_iter += int(
                                ceil(self.coarse_batches / 2 ** (self.coarse_level))
                            )
                            break

                    # copy over the coarse step moments
                    if self.use_coarse_moment:
                        optimizer.param_groups[0]["moment"] = coarse_optim.param_groups[
                            0
                        ]["moment"]

                    # update top level weights
                    def grad_closure():
                        data = next(iter(self.trainloader))
                        data.to(self.device)
                        out, y = self.model(**get_data(data))
                        return self.model.loss(out, y)["loss"]

                    optimizer.update_weights(grad_closure)

                    # full graph steps after coarse
                    if self.after_steps > 0:
                        for i, data in enumerate(
                            tqdm(
                                trainloader,
                                desc=f"Epoch: {epoch:03}",
                                dynamic_ncols=True,
                                disable=self.quiet,
                                total=self.after_steps,
                            ),
                            start=1,
                        ):
                            data.to(self.device)

                            def closure():
                                out, y = self.model(**get_data(data))
                                return self.model.loss(out, y)["loss"]

                            loss = optimizer.step(closure)
                            k_iter += 1
                            train_loss += loss.detach().item() / self.after_steps
                            optimizer.zero_grad()

                            if i >= self.after_steps:
                                break

                # Validation
                valid_loss = self.validate(self.model)

                if not self.quiet:
                    print(f"Epoch: {epoch:03} | Valid Loss: {valid_loss['loss']}")

                mlflow.log_metrics(
                    {"train/loss": train_loss}
                    | {f"validate/{k}": v for k, v in valid_loss.items()},
                    step=k_iter,
                )

            # DD step
            with torch.no_grad():
                for p in self.model.parameters():
                    dist.broadcast(p, src=0)

            if self.rank == 0:
                higher_state = optimizer.state_dict()
            else:
                higher_state = None

            objects = [higher_state]
            dist.broadcast_object_list(objects, src=0)

            # # in case there are more partitions than processes
            # # WIP: would need to communicate all contributions
            # for i in range(0, self.num_parts, self.world_size):
            #     part_i = i + self.rank
            #     if part_i < self.num_parts:
            #         self.precondition(
            #             model_g=self.model,
            #             higher_state=objects[0],  # type: ignore
            #             i=part_i,
            #             epoch=epoch,
            #         )

            self.precondition(
                model_g=self.model,
                higher_state=objects[0],  # type: ignore
                i=self.rank,
                epoch=epoch,
            )

            k_iter += int(ceil(self.pre_steps / self.num_parts))

            # Contribution combination
            w_avg = deepcopy(self.model.state_dict())
            with torch.no_grad():
                for p in w_avg.values():
                    if p.data.dtype == torch.float:
                        dist.reduce(p, op=dist.ReduceOp.AVG, dst=0)

            self.model.load_state_dict(w_avg)

        return valid_loss["loss"]
