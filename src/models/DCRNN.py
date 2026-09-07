import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

from .common import GNN


class DCRNNModel(GNN):
    def __init__(self, K=2, num_layers=2, **kwargs):
        super().__init__(**kwargs)

        self.recurrent = nn.ModuleList(
            [
                DCRNN(
                    in_channels=self.in_dim if i == 0 else self.hidden_dim,
                    out_channels=self.hidden_dim,
                    K=K,
                )
                for i in range(num_layers)
            ]
        )

        # predict all future horizons at once
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, y, edge_index, edge_attr, **_):
        hidden = [None] * len(self.recurrent)

        # consume the history
        for t in range(x.size(-1)):
            h = x[:, :, t]

            for i, layer in enumerate(self.recurrent):
                h = layer(
                    h,
                    edge_index,
                    edge_attr,
                    H=hidden[i],
                )
                hidden[i] = h

        h = F.relu(h)  # type: ignore

        pred = self.linear(h)

        return pred, y

    def loss(self, pred, label) -> dict[str, torch.Tensor]:
        """Masked MAE"""

        mask = (label != 0).float()
        mask /= mask.mean()
        loss = torch.abs(pred - label)
        loss = loss * mask

        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        return {"loss": loss.mean()}
