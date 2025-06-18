import random
from datetime import datetime
from logging import warning
from pathlib import Path
from typing import Callable, Type

import hydra
import matplotlib.pyplot as plt
import mlflow
import networkx as nx
import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig

# from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from models import GCN_CG, GCN_CN
from trainers import GAMMA_ALGO, Accumulating, Batched, Preconditioned, Trainer
from utils import partition_transform_global, position_transform

MODELS = {
    "GCN_CG": GCN_CG,
    "GCN_CN": GCN_CN,
    # "GCN_WikiCS": lambda **kwargs: GCN_CG(hidden_dim=120, out_dim=120, **kwargs),
}

TRAINERS: dict[str, Type[Trainer] | Callable[..., Trainer]] = {
    "batched": Batched,  # baseline
    "accumulating": Accumulating,  # baseline with gradient accumulation
    "pre-accumulating": Preconditioned,
    "pre-batched": lambda **kwargs: Preconditioned(batched=True, **kwargs),
}


def load_data(dataset: str, preprocessing, reload: bool, root: str):
    trainset = GNNBenchmarkDataset(
        root=root,
        name=dataset,
        split="train",
        pre_transform=preprocessing,
        force_reload=reload,
    )

    validset = GNNBenchmarkDataset(
        root=root,
        name=dataset,
        split="val",
        pre_transform=preprocessing,
        force_reload=reload,
    )

    testset = GNNBenchmarkDataset(
        root=root,
        name=dataset,
        split="test",
        pre_transform=preprocessing,
        force_reload=reload,
    )

    return trainset, validset, testset


@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig):
    dataset_dir = Path(cfg.dev.data_dir)

    if not torch.cuda.is_available():
        warning("No CUDA detected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.seed)

    torch.backends.cudnn.benchmark = False

    trainset, validset, testset = load_data(
        cfg.dataset,
        None if cfg.dataset == "PATTERN" else position_transform,
        reload=cfg.u,
        root=str(dataset_dir),
    )
    trainloader = DataLoader(
        trainset,
        batch_size=cfg.dev.batch,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.dev.num_workers,
    )
    validloader = DataLoader(
        validset,
        batch_size=cfg.dev.batch,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.dev.num_workers,
    )
    testloader = DataLoader(
        testset,
        batch_size=cfg.dev.batch,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.dev.num_workers,
    )

    part_trainloader = None
    has_pre = cfg.partitions > 1
    if has_pre:
        partset = GNNBenchmarkDataset(
            root=str(dataset_dir / f"partitioned_{cfg.partitions}"),
            name=cfg.dataset,
            split="train",
            pre_transform=lambda data: partition_transform_global(
                data if cfg.dataset == "PATTERN" else position_transform(data),
                cfg.partitions,
            ),
            force_reload=cfg.u,
        )
        part_trainloader = DataLoader(
            partset,
            batch_size=cfg.dev.batch,
            shuffle=True,
            # NOTE: results in x_0_batch
            follow_batch=[f"x_{i}" for i in range(cfg.partitions)],
            pin_memory=True,
            num_workers=cfg.dev.num_workers,
        )

        # Partitioned graph plotting
        if False:
            for d, data in enumerate(part_trainloader):
                plt.figure()

                for i in range(cfg.partitions):
                    x = data.get_x(i, device)
                    edge_index = data.get_edge_index(i, device)

                    N = x.shape[0]
                    A = torch.zeros((N, N), dtype=torch.float)  # adjacency matrix

                    # bidirectional adjacency matrix
                    A[edge_index[0], edge_index[1]] = 1
                    A[edge_index[1], edge_index[0]] = 1

                    G = nx.from_numpy_array(A.numpy())
                    pos = {
                        node: (x[node, -2].item(), x[node, -1].item())
                        for node in range(N)
                    }

                    rgb = x[:, :3].clamp(0, 1).cpu().numpy()  # Shape: (N, 3)

                    nx.draw(
                        G,
                        pos,
                        node_size=50,
                        node_color=rgb,
                        edge_color="gray",
                        with_labels=False,
                    )

                plt.savefig(f"graphs/graph{d}.png", dpi=300)

                if d == 5:
                    break
            return

    # learning rates and weight decays
    # LRnWDs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    # ruff: noqa: E712
    search_space = {
        "lr": cfg.model.lr,  # if cfg.model.lr != True else hp.choice("lr", LRnWDs),
        "wd": cfg.model.wd,  # if cfg.model.wd != True else hp.choice("wd", LRnWDs),
        **(
            {
                # "pre_epochs": cfg.pre_epochs,
                # "full_epochs": cfg.full_epochs,
                "pre_epochs": hp.choice("pre_epochs", [10, 20, 30, 40]),
                "full_epochs": hp.choice("full_epochs", [1, 3, 5]),
                # if cfg.pre_epochs != True
                # else 5 * hp.uniformint("pre_epochs", 1, 8),
                "pre_lr": cfg.model.pre_lr,
                # if args.pre_lr is not None
                # else hp.choice("pre_lr", LRnWDs),
                "pre_wd": cfg.model.pre_wd,
                # if cfg.pre_wd is not None
                # else hp.choice("pre_wd", LRnWDs),
            }
            if has_pre
            else {}
        ),
    }

    name = ""
    if cfg.name:
        name += f"{cfg.name}_"
    if has_pre:
        if cfg.ASM:
            name += "AS_"  # Additive
        else:
            name += "MS_"  # Multiplicative
    name += f"P{cfg.partitions}_S{cfg.seed}_{cfg.trainer}"

    def objective(params):
        model = MODELS[cfg.model.base](
            in_dim=trainset.num_features,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            n_classes=trainset.num_classes,
            dropout=cfg.model.dropout,
        )

        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params(
                {
                    "seed": cfg.seed,
                    "trainer": cfg.trainer,
                    "model": cfg.model.base,
                    "dataset": cfg.dataset,
                    "batch": cfg.dev.batch,
                    "epochs": cfg.epochs,
                    "additive": cfg.ASM,
                    "line search": cfg.gamma_algo,
                    "partitions": cfg.partitions,
                    **params,
                }
            )
            trainer = TRAINERS[cfg.trainer](
                name=name,
                model=model,
                trainloader=trainloader,
                validloader=validloader,
                testloader=testloader,
                device=device,
                quiet=cfg.dev.q,
                part_trainloader=part_trainloader,
                num_parts=cfg.partitions,
                ASM=cfg.ASM,
                epochs=cfg.epochs,
                gamma_algo=GAMMA_ALGO(cfg.gamma_algo),
                **params,
            )
            loss = trainer.run()

            mlflow.pytorch.log_model(trainer.model, "model")
        return {"loss": loss, "status": STATUS_OK}

    # TODO: add hardware metrics
    mlflow.set_experiment("GNN_" + datetime.now().strftime("%yw%V"))
    with mlflow.start_run(run_name=f"{cfg.dataset}_{name}"):
        trials = Trials()
        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=cfg.max_evals,
            trials=trials,
            show_progressbar=False,
        )

        # if best is not None:
        #     mlflow.log_params(best)


if __name__ == "__main__":
    try:
        # with profile(
        #     activities=[ProfilerActivity.CPU],
        #     on_trace_ready=tensorboard_trace_handler("./log"),
        #     with_stack=True,  # OOM
        # ) as p:
        #     main()
        # # print(p.key_averages().table())

        main()
    except KeyboardInterrupt:
        print("Stopping...")
