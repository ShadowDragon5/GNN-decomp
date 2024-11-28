import argparse
from os import makedirs

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GNNBenchmarkDataset

from models import GCN, SimpleGCN

DATASET_DIR = "./datasets"

MODELS = {
    "GCN": GCN,
    "Simple": SimpleGCN,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", action="store_true")  # verbose
    parser.add_argument("-model", choices=MODELS.keys(), default="GCN")

    parser.add_argument("-seed", type=int, default=42)

    parser.add_argument("-batch", type=int, default=32)
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
    model.to(device)
    model_name = type(model).__name__

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)

            out = model(x, edge_index, batch)
            loss = model.loss(out, data.y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        torch.save(model.state_dict(), f"saves/training_{model_name}_{epoch:03}.pt")

        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for data in validloader:
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                batch = data.batch.to(device)

                out = model(x, edge_index, batch)

                valid_loss += loss.item()

        print(
            f"{model_name} Epoch: {epoch:03} | "
            f"Valid Loss: {valid_loss / len(validloader)}"
        )


def main(args: argparse.Namespace):
    makedirs("saves", exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE: can lead to speedup if input size is static
    torch.backends.cudnn.benchmark = True

    trainset = GNNBenchmarkDataset(root=DATASET_DIR, name="CIFAR10", split="train")
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)

    validset = GNNBenchmarkDataset(root=DATASET_DIR, name="CIFAR10", split="val")
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

    # dataset = Planetoid(root=DATASET_DIR, name="Cora")
    # loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
    # data = dataset[0].to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # model.train()
    # for epoch in range(200):
    #     optimizer.zero_grad()
    #     out = model(data)
    #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #     loss.backward()
    #     optimizer.step()

    # model.eval()
    # pred = model(data).argmax(dim=1)
    # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    # acc = int(correct) / int(data.test_mask.sum())
    # if args.verbose:
    #     print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
