import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

from .common import GNN


class DCRNNModel(GNN):
    def __init__(self, K=2, **kwargs):
        super().__init__(**kwargs)
        self.recurrent = DCRNN(
            in_channels=self.in_dim, out_channels=self.hidden_dim, K=K
        )
        self.linear = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        return self.linear(h)

    def loss(self, pred, label) -> dict[str, torch.Tensor]:
        """Masked MAE"""
        mask = (label != 0).float()
        mask /= mask.mean()
        loss = torch.abs(pred - label)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        return loss.mean()
