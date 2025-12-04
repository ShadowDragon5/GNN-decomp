from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Sequential


class ResidualGCNLayer(nn.Module):
    def __init__(self, dims, dropout):
        super().__init__()

        self.conv = Sequential(
            "x, edge_index",
            [
                (GCNConv(dims, dims), "x, edge_index -> x"),
                nn.BatchNorm1d(dims),
                nn.ReLU(inplace=True),
            ],
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        residual = x
        x = self.conv(x, edge_index)

        # has to be explicit
        x = x + residual

        x = self.dropout(x)
        return x


class GNN(nn.Module, ABC):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        device,
        **_,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim  # TODO: rename
        self.dropout = dropout
        self.device = device

    @abstractmethod
    def loss(self, pred, label) -> dict[str, torch.Tensor]:
        pass
