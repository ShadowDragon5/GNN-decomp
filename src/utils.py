import csv

import mlflow
import torch
from torch_geometric.data import Data


def position_transform(data: Data):
    """Concatenates features and position"""
    x = torch.cat((torch.Tensor(data.x), torch.Tensor(data.pos)), 1)
    return Data(
        x=x,
        y=data.y,
        edge_index=data.edge_index,
        batch=data.batch,
    )


def write_results(
    filename: str,
    epoch=None,
    train_loss=None,
    valid_loss=None,
    valid_acc=None,
    test_acc=None,
):
    if train_loss and valid_loss and valid_acc:
        mlflow.log_metrics(
            {
                "train/loss": train_loss,
                "validate/loss": valid_loss,
                "validate/accuracy": valid_acc,
            },
            step=epoch,
        )
    if test_acc:
        mlflow.log_metric("test/accuracy", test_acc)

    with open(f"results/{filename}", "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, train_loss, valid_loss, valid_acc, test_acc])
