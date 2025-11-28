import torch
import torch.nn as nn
from torch_geometric.nn import Linear, SAGEConv, Sequential

from .common import GNN


class SAGEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = Sequential(
            "x, edge_index",
            [
                (SAGEConv(in_channels, out_channels), "x, edge_index -> x"),
                nn.BatchNorm1d(out_channels, track_running_stats=False),
                nn.ReLU(inplace=True),
            ],
        )

    def forward(self, x, edge_index):
        return self.block(x, edge_index)


class MLP(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()

        self.MLP_layer = nn.Sequential(
            *[
                layer
                for dims in zip(channels[:-2], channels[1:-1])
                for layer in (Linear(*dims), nn.ReLU(inplace=True))
            ],
            Linear(channels[-2], channels[-1]),
        )

    def forward(self, x):
        return self.MLP_layer(x)


class GraphSAGE(GNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # MLP encoder
        self.encoder = MLP([7, 64, 64, 8])

        self.conv = Sequential(
            "x, edge_index",
            [
                # hparams["encoder"][-1]
                (SAGEBlock(8, self.hidden_dim), "x, edge_index -> x"),
                *[
                    (SAGEBlock(self.hidden_dim, self.hidden_dim), "x, edge_index -> x")
                    for _ in range(2)  # nb_hidden_layers-1
                ],
                # hparams["decoder"][0]
                (SAGEConv(self.hidden_dim, 8), "x, edge_index -> x"),
            ],
        )

        # MLP decoder
        self.decoder = MLP([8, 64, 64, 4])

    def forward(self, x, edge_index):
        z = self.encoder(x)

        z = self.conv(z, edge_index)

        z = self.decoder(z)

        return z

    def loss(self, pred, label) -> torch.Tensor:
        loss_criterion = nn.MSELoss(reduction="none")
        return loss_criterion(pred, label).mean()
