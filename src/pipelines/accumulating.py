"""
Full graph, gradient accumulation variation
"""

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from pipelines.common import test
from utils import write_results


def train(
    name: str,
    model: torch.nn.Module,
    trainloader: DataLoader,
    validloader: DataLoader,
    testloader: DataLoader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    quiet: bool = False,
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

        for data in tqdm(
            trainloader,
            desc=f"Epoch: {epoch:02}",
            dynamic_ncols=True,
            disable=quiet,
        ):
            x = data.x.to(device)
            y = data.y.to(device)

            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)

            out = model(x, edge_index, batch)
            loss = model.loss(out, y)

            train_loss += loss.detach().item()

            loss = loss / len(trainloader)
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_loss /= len(trainloader)

        # if epoch == epochs - 1:
        #     torch.save(model.state_dict(), f"saves/training_acc_{name}_{epoch:03}.pt")

        # Validation
        valid_loss = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in tqdm(validloader, dynamic_ncols=True, disable=quiet):
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

        if not quiet:
            print(f"{name} Epoch: {epoch:03} | " f"Valid Loss: {valid_loss}")

        write_results(
            f"{name}_acc.csv",
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            valid_acc=correct / total,
        )

    accuracy = test(model, testloader, device)
    if not quiet:
        print(f"{name} Accuracy: {accuracy}")

    write_results(
        f"{name}_acc.csv",
        test_acc=accuracy,
    )
