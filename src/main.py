import argparse
from os import makedirs
import csv

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GNNBenchmarkDataset
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
    epoch: int,
    train_loss: float,
    valid_loss: float,
):
    with open(f"results/{filename}", "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, train_loss, valid_loss])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", action="store_true")  # verbose
    parser.add_argument("-model", choices=MODELS.keys(), default="GCN")
    parser.add_argument("-dataset", choices=DATASETS, default="MNIST")

    parser.add_argument("-seed", type=int, default=42)

    parser.add_argument("-batch", type=int, default=128)
    parser.add_argument("-epochs", type=int, default=100)

    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-weight_decay", type=float, default=5e-4)

    return parser.parse_args()


def train(
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
        verbose=True,
    )

    model.to(device)
    model_name = type(model).__name__

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

        torch.save(model.state_dict(), f"saves/training_{model_name}_{epoch:03}.pt")

        # Validation
        valid_loss = 0
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
        valid_loss /= len(validloader)

        scheduler.step(valid_loss)
        print(f"{model_name} Epoch: {epoch:03} | " f"Valid Loss: {valid_loss}")

        write_results(
            f"{model_name}.csv",
            epoch,
            train_loss,
            valid_loss,
        )


def main():
    args = parse_arguments()

    makedirs("saves", exist_ok=True)
    makedirs("results", exist_ok=True)

    torch.manual_seed(args.seed)

    # NOTE:
    if not torch.cuda.is_available():
        raise Exception("No CUDA detected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = False

    # Data Prep
    trainset = GNNBenchmarkDataset(root=DATASET_DIR, name=args.dataset, split="train")
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)

    validset = GNNBenchmarkDataset(root=DATASET_DIR, name=args.dataset, split="val")
    validloader = DataLoader(validset, batch_size=args.batch, shuffle=True)

    model = MODELS[args.model](
        in_dim=trainset.num_node_features,
        hidden_dim=146,
        out_dim=146,
        n_classes=trainset.num_classes,
    )

    train(
        model,
        trainloader,
        validloader,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
    )

    testset = GNNBenchmarkDataset(root=DATASET_DIR, name=args.dataset, split="test")
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=True)

    # model.load_state_dict(torch.load("saves/training_SimpleGCN_099.pt", device))
    # model.to(device)

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)

            out = model(x, edge_index, batch)
            pred = out.argmax(dim=1)  # Predicted labels

            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    print(f"Accuracy: {correct / total}")


if __name__ == "__main__":
    main()
