import argparse
import random
from logging import warning
from os import makedirs
from pathlib import Path
from typing import Callable, Type

import mlflow
import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from models import GCN, GCN_NODE
from pipelines.accumulating import Accumulating
from pipelines.batched import Batched
from pipelines.common import Pipeline
from pipelines.pre_accumulating import PreAccumulating
from utils import partition_transform_global, position_transform

DATASET_DIR = Path("./datasets")

MODELS = {
    "GCN_SuperPix": lambda **kwargs: GCN(hidden_dim=146, out_dim=146, **kwargs),
    "GCN_Pattern": lambda **kwargs: GCN_NODE(hidden_dim=146, out_dim=146, **kwargs),
    "GCN_WikiCS": lambda **kwargs: GCN(hidden_dim=120, out_dim=120, **kwargs),
}

PIPELINES: dict[str, Type[Pipeline] | Callable[..., Pipeline]] = {
    "batched": Batched,  # baseline
    "accumulating": Accumulating,  # baseline with gradient accumulation
    "pre-accumulating": PreAccumulating,
    # "pre-batched": PreBatched,
    "pre-batched": lambda **kwargs: PreAccumulating(batched=True, **kwargs),
}

DATASETS = [
    "PATTERN",
    # "CLUSTER",
    "MNIST",
    "CIFAR10",
    # "TSP",
    # "CSL",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-q", action="store_true", help="supress stdout")
    parser.add_argument(
        "-u",
        action="store_true",
        help="force re-generate dataset files",
    )
    parser.add_argument(
        "-partitions",
        type=int,
        default=1,
        help="use partitioned version of datasets and pipelines",
    )
    parser.add_argument(
        "-max_evals",
        type=int,
        default=5,
        help="max evaluations for hyperopt search",
    )
    parser.add_argument("-name", type=str)
    parser.add_argument("-model", choices=MODELS.keys(), default="GCN_SuperPix")
    parser.add_argument("-dataset", choices=DATASETS, default="CIFAR10")
    parser.add_argument("-pipeline", choices=PIPELINES.keys(), default="batched")

    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-batch", type=int, default=128)
    parser.add_argument("-dropout", type=float, default=0.0)

    parser.add_argument("-epochs", type=int)
    parser.add_argument("-lr", type=float, help="learning rate")
    parser.add_argument("-wd", type=float, help="weight decay")

    parser.add_argument("-pre_epochs", type=int)
    parser.add_argument(
        "-pre_lr",
        type=float,
        help="preconditioner learning rate (only for 'pre-' pipelines)",
    )
    parser.add_argument(
        "-pre_wd",
        type=float,
        help="preconditioner weight decay (only for 'pre-' pipelines)",
    )
    parser.add_argument(
        "-ASM",
        action="store_true",
        help="use Additive Schwarz Method preconditioner",
    )

    return parser.parse_args()


def load_data(dataset: str, preprocessing, reload: bool):
    trainset = GNNBenchmarkDataset(
        root=str(DATASET_DIR),
        name=dataset,
        split="train",
        pre_transform=preprocessing,
        force_reload=reload,
    )

    validset = GNNBenchmarkDataset(
        root=str(DATASET_DIR),
        name=dataset,
        split="val",
        pre_transform=preprocessing,
        force_reload=reload,
    )

    testset = GNNBenchmarkDataset(
        root=str(DATASET_DIR),
        name=dataset,
        split="test",
        pre_transform=preprocessing,
        force_reload=reload,
    )

    return trainset, validset, testset


def main():
    args = parse_arguments()

    makedirs("saves", exist_ok=True)
    makedirs("results", exist_ok=True)

    if not torch.cuda.is_available():
        warning("No CUDA detected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False

    trainset, validset, testset = load_data(
        args.dataset,
        None if args.dataset == "PATTERN" else position_transform,
        reload=args.u,
    )
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)

    part_trainloader = None
    has_pre = args.partitions > 1
    if has_pre:
        partset = GNNBenchmarkDataset(
            root=str(DATASET_DIR / f"partitioned_{args.partitions}"),
            name=args.dataset,
            split="train",
            pre_transform=lambda data: partition_transform_global(
                data if args.dataset == "PATTERN" else position_transform(data),
                args.partitions,
            ),
            force_reload=args.u,
        )
        part_trainloader = DataLoader(
            partset,
            batch_size=args.batch,
            shuffle=True,
            follow_batch=[f"x_{i}" for i in range(args.partitions)],
        )

    search_space = {
        "epochs": args.epochs
        if args.epochs is not None
        else hp.uniformint("epochs", 10, 100),
        "lr": args.lr
        if args.lr is not None
        else hp.loguniform("lr", np.log(1e-5), np.log(1e-2)),
        "wd": args.wd
        if args.wd is not None
        else hp.loguniform("wd", np.log(1e-5), np.log(1e-2)),
        **(
            {
                "pre_epochs": args.pre_epochs
                if args.pre_epochs is not None
                else hp.uniformint("pre_epochs", 1, 100),
                "pre_lr": args.pre_lr
                if args.pre_lr is not None
                else hp.loguniform("pre_lr", np.log(1e-6), np.log(1e-2)),
                "pre_wd": args.pre_wd
                if args.pre_wd is not None
                else hp.loguniform("pre_wd", np.log(1e-5), np.log(1e-3)),
            }
            if has_pre
            else {}
        ),
    }

    name = ""
    if args.name:
        name += f"{args.name}_"
    if has_pre:
        if args.ASM:
            name += "AS_"  # Additive
        else:
            name += "MS_"  # Multiplicative
    name += f"{args.model}_p{args.partitions}_s{args.seed}"

    def objective(params):
        model = MODELS[args.model](
            in_dim=trainset.num_features,
            n_classes=trainset.num_classes,
            dropout=args.dropout,
        )

        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params(params)
            pipeline = PIPELINES[args.pipeline](
                name=name,
                model=model,
                trainloader=trainloader,
                validloader=validloader,
                testloader=testloader,
                device=device,
                quiet=args.q,
                part_trainloader=part_trainloader,
                num_parts=args.partitions,
                ASM=args.ASM,
                **params,
            )
            loss = pipeline.run()

            mlflow.pytorch.log_model(pipeline.model, "model")
        return {"loss": loss, "status": STATUS_OK}

    mlflow.set_experiment("graph-partitioning")
    with mlflow.start_run(run_name=f"{args.dataset}_{name}"):
        mlflow.log_params(
            {
                "seed": args.seed,
                "pipeline": args.pipeline,
                "model": args.model,
                "dataset": args.dataset,
                "batch": args.batch,
            }
        )
        trials = Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=args.max_evals,
            trials=trials,
        )

        if best is not None:
            mlflow.log_params(best)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping...")
