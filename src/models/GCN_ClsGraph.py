import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, global_mean_pool

from .common import ResidualGCNLayer


class GCN_CG(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_classes: int,
        dropout=0.0,
    ):
        super().__init__()

        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        self.conv = Sequential(
            "x, edge_index",
            [
                (ResidualGCNLayer(hidden_dim, dropout), "x, edge_index -> x"),
                (ResidualGCNLayer(hidden_dim, dropout), "x, edge_index -> x"),
                (ResidualGCNLayer(hidden_dim, dropout), "x, edge_index -> x"),
                (ResidualGCNLayer(hidden_dim, dropout), "x, edge_index -> x"),
            ],
        )

        self.MLP_layer = nn.Sequential(
            nn.Linear(out_dim, out_dim >> 1),
            nn.ReLU(),
            nn.Linear(out_dim >> 1, out_dim >> 2),
            nn.ReLU(),
            nn.Linear(out_dim >> 2, n_classes),
        )

    def forward(self, x, edge_index, batch):
        h = self.embedding_h(x)

        h = self.conv(h, edge_index)

        # per node features [n, f] -> graph features [n_batches, f]
        h = global_mean_pool(h, batch)

        h = self.MLP_layer(h)
        return h

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        return criterion(pred, label)


if __name__ == "__main__":
    model = GCN_CG(
        hidden_dim=146,
        out_dim=146,
        in_dim=5,
        n_classes=2,
    )

    d = model.state_dict()
    for k, v in d.items():
        print(v.data.dtype)
        if v.data.dtype != torch.float:
            print(k, v)
