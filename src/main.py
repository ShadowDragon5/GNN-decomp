import argparse
import random
from logging import warning
from os import makedirs
from pathlib import Path
from typing import Type

import mlflow
import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from models import GCN, SimpleGCN
from pipelines.accumulating import Accumulating
from pipelines.batched import Batched
from pipelines.common import Pipeline
from pipelines.pre_accumulating import PreAccumulating
from pipelines.pre_batched import PreBatched
from utils import partition_transform, position_transform

DATASET_DIR = Path("./datasets")

MODELS = {
    "GCN": GCN,
    "Simple": SimpleGCN,
}

PIPELINES: dict[str, Type[Pipeline]] = {
    "batched": Batched,  # baseline
    "accumulating": Accumulating,  # baseline with gradient accumulation
    "pre-accumulating": PreAccumulating,
    "pre-batched": PreBatched,
}

DATASETS = [
    # "PATTERN",
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
    parser.add_argument("-model", choices=MODELS.keys(), default="GCN")
    parser.add_argument("-dataset", choices=DATASETS, default="CIFAR10")
    parser.add_argument("-pipeline", choices=PIPELINES.keys(), default="batched")

    parser.add_argument("-seed", type=int, default=42)

    parser.add_argument("-batch", type=int, default=128)
    parser.add_argument("-epochs", type=int, default=100)

    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "-pre_lr",
        type=float,
        default=0.0001,
        help="preconditioner learning rate (only for 'pre-' pipelines)",
    )
    parser.add_argument("-weight_decay", type=float, default=5e-4)

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
        args.dataset, position_transform, reload=args.u
    )
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)

    part_trainloader = None
    if args.partitions > 1:
        partset = GNNBenchmarkDataset(
            root=str(DATASET_DIR / f"partitioned_{args.partitions}"),
            name=args.dataset,
            split="train",
            pre_transform=lambda data: partition_transform(data, args.partitions),
            # force_reload=True,  # args.u,
        )
        part_trainloader = DataLoader(
            partset,
            batch_size=args.batch,
            shuffle=True,
            follow_batch=[f"x_{i}" for i in range(args.partitions)],
        )

    model = MODELS[args.model](
        in_dim=trainset.num_features,
        hidden_dim=146,
        out_dim=146,
        n_classes=trainset.num_classes,
    )

    name = f"{type(model).__name__}_p{args.partitions}_seed{args.seed}"

    params = {
        "epochs": args.epochs,
        "pre_epochs": 1,
        "lr": args.lr,
        "pre_lr": args.pre_lr,
        "wd": args.weight_decay,
        "pre_wd": 0,
    }

    search_space = {
        "epochs": hp.uniformint("epochs", 100, 200),
        "pre_epochs": hp.uniformint("pre_epochs", 1, 100),
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
        "pre_lr": hp.loguniform("pre_lr", np.log(1e-6), np.log(1e-2)),
        "wd": hp.loguniform("wd", np.log(1e-5), np.log(1e-1)),
        "pre_wd": hp.loguniform("pre_wd", np.log(1e-5), np.log(1e-1)),
    }

    def objective(params):
        l_name = name + f"_lr{params['lr']}_prec_lr{params['pre_lr']}"
        print(l_name)
        with mlflow.start_run(run_name=l_name, nested=True):
            loss = PIPELINES[args.pipeline](
                name=l_name,
                model=model,
                trainloader=trainloader,
                validloader=validloader,
                testloader=testloader,
                device=device,
                quiet=args.q,
                part_trainloader=part_trainloader,
                num_parts=args.partitions,
                **params,
            ).run()

            mlflow.pytorch.log_model(model, l_name)
        return {"loss": loss, "status": STATUS_OK}

    mlflow.set_experiment("graph-partitioning")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "seed": args.seed,
                "pipeline": args.pipeline,
                "model": args.model,
                "dataset": args.dataset,
                "batch": args.batch,
                **params,
            }
        )
        trials = Trials()
        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping...")
