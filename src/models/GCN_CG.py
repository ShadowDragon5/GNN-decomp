import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, global_mean_pool

from .common import GNN, ResidualGCNLayer


class GCN_CG(GNN):
    def __init__(self, n_classes: int, **kwargs):
        super().__init__(**kwargs)

        self.embedding_h = nn.Linear(self.in_dim, self.hidden_dim)

        self.conv = Sequential(
            "x, edge_index",
            [
                (ResidualGCNLayer(self.hidden_dim, self.dropout), "x, edge_index -> x"),
                (ResidualGCNLayer(self.hidden_dim, self.dropout), "x, edge_index -> x"),
                (ResidualGCNLayer(self.hidden_dim, self.dropout), "x, edge_index -> x"),
                (ResidualGCNLayer(self.hidden_dim, self.dropout), "x, edge_index -> x"),
            ],
        )

        self.MLP_layer = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim >> 1),
            nn.ReLU(),
            nn.Linear(self.out_dim >> 1, self.out_dim >> 2),
            nn.ReLU(),
            nn.Linear(self.out_dim >> 2, n_classes),
        )

    def forward(self, x, edge_index, batch):
        h = self.embedding_h(x)

        h = self.conv(h, edge_index)

        # per node features [n, f] -> graph features [n_batches, f]
        h = global_mean_pool(h, batch)

        h = self.MLP_layer(h)
        return h

    def loss(self, pred, label) -> torch.Tensor:
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
