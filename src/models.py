import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Sequential, global_mean_pool


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


class GCN(nn.Module):
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


class GCN_NODE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_classes: int,
        dropout=0.0,
    ):
        super().__init__()
        self.n_classes = n_classes

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

    def forward(self, x, edge_index, *_):
        h = self.embedding_h(x)

        h = self.conv(h, edge_index)

        h = self.MLP_layer(h)
        return h

    def loss(self, pred, label):
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(label.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        criterion = nn.CrossEntropyLoss()
        return criterion(pred, label)


if __name__ == "__main__":
    model = GCN(
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
