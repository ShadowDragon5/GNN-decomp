from datetime import datetime
from enum import StrEnum, auto
from logging import warning
from os import makedirs
from pathlib import Path
from typing import Callable, Type

import hydra
import matplotlib.pyplot as plt
import mlflow
import networkx as nx
import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mlflow.pytorch import log_model
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import AirfRANS, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from data import wave_data_2D_irrgular
from models import GCN_CG, GCN_CN, GraphSAGE, MeshGraphNet
from trainers import (
    GAMMA_ALGO,
    WEIGHTING_STRATEGY,
    Accumulating,
    Batched,
    MGN_trainer,
    Preconditioned,
    Trainer,
)
from utils import (
    get_data,
    normalization_transform,
    partition_data_points,
    partition_transform_global,
    position_transform,
)

MODELS = {
    "GCN_CG": GCN_CG,
    "GCN_CN": GCN_CN,  # Pattern
    # "GCN_WikiCS": lambda **kwargs: GCN_CG(hidden_dim=120, out_dim=120, **kwargs),
    "MeshGraphNet": MeshGraphNet,
    "GraphSAGE": GraphSAGE,
}

TRAINERS: dict[str, Type[Trainer] | Callable[..., Trainer]] = {
    "batched": Batched,  # baseline
    "accumulating": Accumulating,  # baseline with gradient accumulation
    "pre-accumulating": Preconditioned,
    "pre-batched": lambda **kwargs: Preconditioned(batched=True, **kwargs),
    "mgn-batched": MGN_trainer,  # MGN baseline
}

SCHEDULERS = {
    "GCN_CG": lambda optim, *_: torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=0.5,
        patience=5,
    ),
    "GraphSAGE": lambda optim, lr, total_steps: torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=lr,
        total_steps=total_steps,
    ),
}

# Gamma optimizer
OPTIM = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "SGDm": lambda **kwargs: torch.optim.SGD(**kwargs, momentum=0.1),
    "RMSprop": torch.optim.RMSprop,
}


class DS(StrEnum):
    """Dataset"""

    @staticmethod
    def _generate_next_value_(name: str, *_, **__) -> str:
        return name

    MNIST = auto()
    CIFAR10 = auto()
    PATTERN = auto()
    Wave2D = auto()
    AirfRANS = auto()


# Normalization variables
mean_x = 0
std_x = 0
mean_y = 0
std_y = 0


def load_data(name: DS, reload: bool, root: Path) -> tuple[Dataset, Dataset, Dataset]:
    """
    name: The name of the dataset
    reload: A flag if a dataset should be downloaded anew
    """

    root_str = str(root)
    match name:
        case DS.CIFAR10 | DS.MNIST | DS.PATTERN:
            preprocessing = None
            if name in [DS.CIFAR10, DS.MNIST]:
                preprocessing = position_transform

            trainset = GNNBenchmarkDataset(
                root=root_str,
                name=name,
                split="train",
                pre_transform=preprocessing,
                force_reload=reload,
            )

            validset = GNNBenchmarkDataset(
                root=root_str,
                name=name,
                split="val",
                pre_transform=preprocessing,
                force_reload=reload,
            )

            testset = GNNBenchmarkDataset(
                root=root_str,
                name=name,
                split="test",
                pre_transform=preprocessing,
                force_reload=reload,
            )

        case DS.Wave2D:
            root_str = str(root / "PDE/training")
            trainset = wave_data_2D_irrgular(
                edge_features=["dist", "direction"],
                endtime=250,
                root=root_str,
                node_features=["u", "v", "density", "type"],
                num_trajectory=1000,
                step_size=5,
                train=True,
                var=0,
                force_reload=reload,
            )

            validset = wave_data_2D_irrgular(
                edge_features=["dist", "direction"],
                endtime=250,
                root=root_str,
                node_features=["u", "v", "density", "type"],
                num_trajectory=1,
                step_size=5,
                train=False,
                var=0,
                force_reload=reload,
            )

            testset = validset

        case DS.AirfRANS:
            root_str = str(root / "AirfRANS")
            # task = "full"
            task = "scarce"
            trainset = AirfRANS(
                root=root_str,
                task=task,
                train=True,
                force_reload=reload,
            )

            global mean_x, mean_y, std_x, std_y
            items = 0
            for data in trainset:
                items += data.x.shape[0]
                mean_x += (data.x.sum(axis=0) - data.x.shape[0] * mean_x) / items
                mean_y += (data.y.sum(axis=0) - data.y.shape[0] * mean_y) / items

            items = 0
            for data in trainset:
                n = data.x.shape[0]
                items += n
                std_x += (((data.x - mean_x) ** 2).sum(axis=0) - n * std_x) / items
                std_y += (((data.y - mean_y) ** 2).sum(axis=0) - n * std_y) / items

            std_x = torch.sqrt(std_x)  # type: ignore
            std_y = torch.sqrt(std_y)  # type: ignore

            trainset = AirfRANS(
                root=root_str,
                task=task,
                train=True,
                pre_transform=lambda data: position_transform(
                    normalization_transform(data, mean_x, std_x, mean_y, std_y)
                ),
                force_reload=reload,
            )

            validset = AirfRANS(
                root=root_str,
                task=task,
                train=False,
                pre_transform=lambda data: position_transform(
                    normalization_transform(data, mean_x, std_x, mean_y, std_y)
                ),
                force_reload=reload,
            )

            testset = validset

    return trainset, validset, testset


@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig):
    makedirs("results", exist_ok=True)
    dataset_dir = Path(cfg.dev.data_dir)

    if not torch.cuda.is_available():
        warning("No CUDA detected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.seed)

    torch.backends.cudnn.benchmark = False

    trainset, validset, testset = load_data(
        DS(cfg.dataset),
        reload=cfg.u,
        root=dataset_dir,
    )
    trainloader = DataLoader(
        trainset,  # type: ignore
        batch_size=cfg.dev.batch,
        shuffle=True,
        num_workers=cfg.dev.num_workers,
    )
    validloader = DataLoader(
        validset,  # type: ignore
        batch_size=cfg.dev.batch,
        shuffle=False,
        num_workers=cfg.dev.num_workers,
    )
    testloader = DataLoader(
        testset,  # type: ignore
        batch_size=cfg.dev.batch,
        shuffle=False,
        num_workers=cfg.dev.num_workers,
    )

    part_trainloader = None
    has_pre = cfg.partitions > 1
    if has_pre:
        partset = None
        root_str = str(dataset_dir / f"partitioned_{cfg.partitions}")
        match cfg.dataset:
            case DS.CIFAR10 | DS.MNIST | DS.PATTERN:
                partset = GNNBenchmarkDataset(
                    root=root_str,
                    name=cfg.dataset,
                    split="train",
                    pre_transform=lambda data: partition_transform_global(
                        data if cfg.dataset == DS.PATTERN else position_transform(data),
                        cfg.partitions,
                    ),
                    force_reload=cfg.u,
                )
            case DS.AirfRANS:
                partset = AirfRANS(
                    root=str(
                        dataset_dir / f"partitioned_{cfg.partitions}" / "AirfRANS"
                    ),
                    task="scarce",
                    train=True,
                    pre_transform=lambda data: partition_data_points(
                        position_transform(
                            normalization_transform(data, mean_x, std_x, mean_y, std_y)
                        ),
                        cfg.partitions,
                    ),
                    force_reload=cfg.u,
                )

            case DS.Wave2D:
                partset = wave_data_2D_irrgular(
                    edge_features=["dist", "direction"],
                    endtime=250,
                    root=str(dataset_dir / f"PDE/partitioned_{cfg.partitions}"),
                    node_features=["u", "v", "density", "type"],
                    num_trajectory=1000,
                    step_size=5,
                    train=True,
                    var=0,
                    pre_transform=lambda data: partition_transform_global(
                        data, cfg.partitions
                    ),
                    force_reload=cfg.u,
                )
        assert partset is not None

        part_trainloader = DataLoader(
            partset,
            batch_size=cfg.dev.batch,
            shuffle=True,
            # NOTE: results in x_0_batch
            follow_batch=[f"x_{i}" for i in range(cfg.partitions)],
            num_workers=cfg.dev.num_workers,
        )

        # Partitioned graph plotting
        if False:
            for d, data in enumerate(part_trainloader):
                plt.figure()

                for i in range(cfg.partitions):
                    # x = data.get("x", i, device)
                    # # edge_index = data.get("edge_index", i, device)
                    #
                    # edge_index = radius_graph(
                    #     x=data.get("pos", i, device),
                    #     r=0.05,
                    #     loop=True,
                    #     max_num_neighbors=64,
                    # )

                    sample_data = get_data(data, i, device)
                    x = sample_data["x"]
                    edge_index = sample_data["edge_index"]
                    pos = sample_data["pos"]

                    N = x.shape[0]
                    G = to_networkx(
                        Data(x=x, edge_index=edge_index),
                        to_undirected=True,
                    )
                    pos = {
                        # node: (x[node, -2].item(), x[node, -1].item())  # CIFAR
                        # node: data.get("coords", i, device)[node]  # Wave2D
                        # node: data.get("pos", i, device)[node]  # AirfRANS
                        node: pos[node]
                        for node in range(N)
                    }

                    # rgb = x[:, :3].clamp(0, 1).cpu().numpy()  # Shape: (N, 3)

                    nx.draw(
                        G,
                        pos,
                        node_size=1,
                        node_color=["steelblue", "crimson"][i],
                        edge_color="gray",
                        with_labels=False,
                    )

                plt.savefig(f"graphs/airfrans/graph{d}.png", dpi=300)

                if d == 5:
                    break
            return

    # learning rates and weight decays
    # LRnWDs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    search_space = {
        "lr": cfg.model.lr,
        "wd": cfg.model.wd,
        **(
            {
                "pre_epochs": cfg.pre_epochs,
                "full_epochs": cfg.full_epochs,
                # "pre_epochs": hp.choice("pre_epochs", [10, 20, 30, 40]),
                # "full_epochs": hp.choice("full_epochs", [1, 3, 5]),
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

    def objective(trainer_params):
        n_classes = None
        if cfg.dataset in [DS.CIFAR10, DS.MNIST, DS.PATTERN]:
            n_classes = trainset.num_classes

        model = MODELS[cfg.model.base](
            in_dim=trainset.num_features,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            edge_dim=3,
            num_steps=10,
            device=device,
            dropout=cfg.model.dropout,
            n_classes=n_classes,
        )

        with mlflow.start_run(
            run_name=f"{cfg.dataset}_{name}",
            description=cfg.description,
        ):
            mlflow.log_params(
                {
                    "seed": cfg.seed,
                    "trainer": cfg.trainer,
                    "optimizer": cfg.optim,
                    "model": cfg.model.base,
                    "hidden_dim": cfg.model.hidden_dim,
                    "dataset": cfg.dataset,
                    "batch": cfg.dev.batch,
                    "epochs": cfg.epochs,
                    "additive": cfg.ASM,
                    "line search": cfg.gamma_algo,
                    "gamma opt. lr": cfg.gamma_lr,
                    "gamma weighting": cfg.gamma_strat,
                    "partitions": cfg.partitions,
                    "optim target": cfg.target,
                    **trainer_params,
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
                target=cfg.target,
                need_acc=cfg.dataset in [DS.CIFAR10, DS.MNIST, DS.PATTERN],
                optim=OPTIM[cfg.optim],
                ll_resolution=cfg.ll_resolution,
                gamma_lr=cfg.gamma_lr,
                gamma_strat=WEIGHTING_STRATEGY(cfg.gamma_strat),
                scheduler=SCHEDULERS[cfg.model.base],
                **trainer_params,
            )
            loss = trainer.run()

            log_model(trainer.model, "model")
        return {"loss": loss, "status": STATUS_OK}

    mlflow.set_experiment("GNN_" + datetime.now().strftime("%yw%V"))
    trials = Trials()
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=cfg.max_evals,
        trials=trials,
        show_progressbar=False,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping...")
