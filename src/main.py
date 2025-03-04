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
from torch_geometric.data import ClusterLoader, Data
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import ClusterData, DataLoader

from models import GCN, SimpleGCN
from pipelines.accumulating import Accumulating
from pipelines.batched import Batched
from pipelines.common import Pipeline
from pipelines.pre_accumulating import PreAccumulating
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

    trainset, validset, testset = load_data(args.dataset, position_transform, args.u)
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)

    part_trainloaders = None
    if args.partitions > 1:
        testset = GNNBenchmarkDataset(
            root=str(DATASET_DIR / f"partitioned_{args.partitions}"),
            name=args.dataset,
            split="train",
            pre_transform=lambda data: partition_transform(data, args.partitions),
            # force_reload=reload,
        )

        # cluster_data = ClusterData(
        #     trainset._data,
        #     num_parts=args.partitions,
        #     save_dir=str(DATASET_DIR / args.dataset),
        #     filename="train.pt",
        # )

        # loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)
        # print(len(loader))
        # it = iter(loader)
        # print(next(it))
        # print(next(it))
        # return

        # part_trainloaders = [
        #     DataLoader(
        #         [
        #             Data(
        #                 x=cluster_data[i].x,
        #                 y=cluster_data[i].y,
        #                 # HACK: all edge indexes are left in the first partition
        #                 edge_index=cluster_data[0].edge_index,
        #             ),
        #         ],
        #         batch_size=args.batch,
        #         shuffle=True,
        #     )
        #     for i in range(len(cluster_data))
        # ]
        # return

        # part_trainloaders = [
        #     DataLoader(
        #         dataset=DatasetPart(
        #             Data(
        #                 x=cluster_data[i].x,
        #                 y=cluster_data[i].y,
        #                 # HACK: all edge indexes are left in the first partition
        #                 edge_index=cluster_data[0].edge_index,
        #             ),
        #             root=str(
        #                 DATASET_DIR / f"{args.dataset}p{args.partitions}_split{i}"
        #             ),
        #         ),
        #         batch_size=args.batch,
        #         shuffle=True,
        #     )
        #     for i in range(len(cluster_data))
        # ]

    model = MODELS[args.model](
        in_dim=trainset.num_features,
        hidden_dim=146,
        out_dim=146,
        n_classes=trainset.num_classes,
    )

    name = f"{type(model).__name__}_p{args.partitions}_seed{args.seed}_lr{args.lr}_prec_lr{args.pre_lr}"

    params = {
        "epochs": args.epochs,
        "lr": args.lr,
        "pre_lr": args.pre_lr,
        "wd": args.weight_decay,
        "pre_wd": 0,
        "pre_epochs": 1,
    }

    # search_space = {
    #         "lr"
    #         }

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
        PIPELINES[args.pipeline](
            name=name,
            model=model,
            trainloader=trainloader,
            validloader=validloader,
            testloader=testloader,
            device=device,
            quiet=args.q,
            part_trainloaders=part_trainloaders,
            **params,
        ).run()

        # mlflow.pytorch.log_model(model, name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping...")
