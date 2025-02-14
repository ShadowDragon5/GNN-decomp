import csv

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
    with open(f"results/{filename}", "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, train_loss, valid_loss, valid_acc, test_acc])
