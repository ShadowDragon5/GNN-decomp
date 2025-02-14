import argparse
import random
from logging import warning
from os import makedirs
from pathlib import Path

# from torch_geometric.distributed import Partitioner
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from models import GCN, SimpleGCN
from pipelines.accumulating import train as accum_train
from pipelines.batched import train as batched_train
from pipelines.full_pass import train as full_train
from utils import position_transform

DATASET_DIR = Path("./datasets")

MODELS = {
    "GCN": GCN,
    "Simple": SimpleGCN,
}

PIPELINES = {
    "batched": batched_train,
    "accumulating": accum_train,
    "full": full_train,
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

    parser.add_argument("-q", action="store_true")  # quiet
    parser.add_argument("-u", action="store_true")  # update data
    parser.add_argument("-model", choices=MODELS.keys(), default="GCN")
    parser.add_argument("-dataset", choices=DATASETS, default="CIFAR10")
    parser.add_argument("-pipeline", choices=PIPELINES.keys(), default="batched")

    parser.add_argument("-seed", type=int, default=42)

    parser.add_argument("-batch", type=int, default=128)
    parser.add_argument("-epochs", type=int, default=100)

    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-weight_decay", type=float, default=5e-4)

    return parser.parse_args()


"""
dont abstact prematurely!
pipelines.batched
pipelines.accumulating
pipelines.accumulating_partitioned
...
"""


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


# print(trainset[0].edge_attr)

# partitioner = Partitioner(
#     trainset._data,
#     num_parts=2,
#     root=DATASET_DIR / "partition",
# )

# partitioner.generate_partition()


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

    # data = next(iter(trainloader))
    # print(data.x[:10])
    # print(data.y[:10])
    # print(trainset.num_features)
    # return

    model = MODELS[args.model](
        in_dim=trainset.num_features,
        hidden_dim=146,
        out_dim=146,
        n_classes=trainset.num_classes,
    )

    name = f"{type(model).__name__}_seed{args.seed}_wd{args.weight_decay}"

    PIPELINES[args.pipeline](
        name,
        model,
        trainloader,
        validloader,
        testloader,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.q,
    )

    # model.load_state_dict(torch.load("saves/training_SimpleGCN_099.pt", device))
    # model.to(device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping...")
