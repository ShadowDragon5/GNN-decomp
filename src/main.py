import argparse
import csv
import random
from os import makedirs

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

# from torch.utils.tensorboard import SummaryWriter
from models import GCN, SimpleGCN

DATASET_DIR = "./datasets"

MODELS = {
    "GCN": GCN,
    "Simple": SimpleGCN,
}

DATASETS = [
    # "PATTERN",
    # "CLUSTER",
    "MNIST",
    "CIFAR10",
    # "TSP",
    # "CSL",
]


def write_results(
    filename: str,
    epoch=None,
    train_loss=None,
    valid_loss=None,
    valid_acc=None,
    test_acc=None,
):
    with open(f"results/{filename}", "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, train_loss, valid_loss, valid_acc, test_acc])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", action="store_true")  # verbose
    parser.add_argument("-model", choices=MODELS.keys(), default="GCN")
    parser.add_argument("-dataset", choices=DATASETS, default="CIFAR10")

    parser.add_argument("-seed", type=int, default=42)

    parser.add_argument("-batch", type=int, default=128)
    parser.add_argument("-epochs", type=int, default=100)

    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-weight_decay", type=float, default=5e-4)

    return parser.parse_args()


def train(
    name: str,
    model: torch.nn.Module,
    trainloader: DataLoader,
    validloader: DataLoader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    model.to(device)

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for data in trainloader:
            x = data.x.to(device)
            y = data.y.to(device)

            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)

            optimizer.zero_grad()
            out = model(x, edge_index, batch)
            loss = model.loss(out, y)

            train_loss += loss.detach().item()

            loss.backward()
            optimizer.step()

        train_loss /= len(trainloader)

        if epoch == epochs - 1:
            torch.save(model.state_dict(), f"saves/training_{name}_{epoch:03}.pt")

        # Validation
        valid_loss = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in validloader:
                x = data.x.to(device)
                y = data.y.to(device)

                edge_index = data.edge_index.to(device)
                batch = data.batch.to(device)

                out = model(x, edge_index, batch)
                loss = model.loss(out, y)
                valid_loss += loss.detach().item()

                # Validation accuracy
                pred = out.argmax(dim=1)  # Predicted labels
                correct += (pred == y).sum().item()
                total += y.size(0)

        valid_loss /= len(validloader)

        scheduler.step(valid_loss)
        # print(f"{name} Epoch: {epoch:03} | " f"Valid Loss: {valid_loss}")

        write_results(
            f"{name}.csv",
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            valid_acc=correct / total,
        )


def position_transform(data: Data):
    """Concatenates features and position"""
    data.x = torch.cat((torch.Tensor(data.x), torch.Tensor(data.pos)), 1)
    return data


def main():
    args = parse_arguments()

    makedirs("saves", exist_ok=True)
    makedirs("results", exist_ok=True)

    # HACK:
    if not torch.cuda.is_available():
        raise Exception("No CUDA detected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False

    # Data Prep
    trainset = GNNBenchmarkDataset(
        root=DATASET_DIR,
        name=args.dataset,
        split="train",
        pre_transform=position_transform,
        # force_reload=True,
    )
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)

    # data = next(iter(trainloader))
    # print(data.x[:10])
    # print(data.y[:10])
    # print(trainset.num_features)
    # return

    validset = GNNBenchmarkDataset(
        root=DATASET_DIR,
        name=args.dataset,
        split="val",
        pre_transform=position_transform,
        # force_reload=True,
    )
    validloader = DataLoader(validset, batch_size=args.batch, shuffle=False)

    model = MODELS[args.model](
        in_dim=trainset.num_features,
        hidden_dim=146,
        out_dim=146,
        n_classes=trainset.num_classes,
    )

    name = f"{type(model).__name__}_seed{args.seed}_wd{args.weight_decay}"

    train(
        name,
        model,
        trainloader,
        validloader,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
    )

    testset = GNNBenchmarkDataset(
        root=DATASET_DIR,
        name=args.dataset,
        split="test",
        pre_transform=position_transform,
        # force_reload=True,
    )
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)

    # model.load_state_dict(torch.load("saves/training_SimpleGCN_099.pt", device))
    # model.to(device)

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            x = data.x.to(device)
            y = data.y.to(device)

            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)

            out = model(x, edge_index, batch)
            pred = out.argmax(dim=1)  # Predicted labels

            correct += (pred == y).sum().item()
            total += y.size(0)

    print(f"{name} Accuracy: {correct / total}")

    write_results(
        f"{name}.csv",
        test_acc=correct / total,
    )


if __name__ == "__main__":
    main()
