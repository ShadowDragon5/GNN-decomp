import random
from datetime import datetime
from logging import warning
from pathlib import Path
from typing import Callable, Type

import hydra
import mlflow
import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from models import GCN_CG, GCN_CN
from trainers import Accumulating, Batched, Preconditioned, Trainer
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

DATASETS = [
    "PATTERN",
    # "CLUSTER",
    "MNIST",
    "CIFAR10",
    # "TSP",
    # "CSL",
]


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
    trainloader = DataLoader(trainset, batch_size=cfg.dev.batch, shuffle=True)
    validloader = DataLoader(validset, batch_size=cfg.dev.batch, shuffle=False)
    testloader = DataLoader(testset, batch_size=cfg.dev.batch, shuffle=False)

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
            # FIX: results in x_0_batch
            follow_batch=[f"x_{i}" for i in range(cfg.partitions)],
        )

    # learning rates and weight decays
    LRnWDs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    # ruff: noqa: E712
    search_space = {
        "lr": cfg.model.lr if cfg.model.lr != True else hp.choice("lr", LRnWDs),
        "wd": cfg.model.wd if cfg.model.wd != True else hp.choice("wd", LRnWDs),
        **(
            {
                "pre_epochs": cfg.pre_epochs
                if cfg.pre_epochs != True
                else 5 * hp.uniformint("pre_epochs", 1, 8),
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
                quiet=cfg.q,
                part_trainloader=part_trainloader,
                num_parts=cfg.partitions,
                ASM=cfg.ASM,
                epochs=cfg.epochs,
                **params,
            )
            loss = trainer.run()

            # mlflow.pytorch.log_model(trainer.model, "model")
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
        main()
    except KeyboardInterrupt:
        print("Stopping...")
