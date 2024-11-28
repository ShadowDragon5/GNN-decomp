import torch.nn as nn
from torch_geometric.nn import GCNConv, Sequential, global_mean_pool


class ResidualGCNLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv = Sequential(
            "x, edge_index",
            [
                (GCNConv(dims, dims), "x, edge_index -> x"),
                nn.BatchNorm1d(dims),
                nn.ReLU(inplace=True),
            ],
        )

    def forward(self, x, edge_index):
        residual = x
        x = self.conv(x, edge_index)

        # has to be explicit
        x = x + residual

        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_classes):
        super().__init__()

        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        self.conv = Sequential(
            "x, edge_index",
            [
                (ResidualGCNLayer(hidden_dim), "x, edge_index -> x"),
                (ResidualGCNLayer(hidden_dim), "x, edge_index -> x"),
                (ResidualGCNLayer(hidden_dim), "x, edge_index -> x"),
                (ResidualGCNLayer(hidden_dim), "x, edge_index -> x"),
            ],
        )

        self.MLP_layer = nn.Sequential(
            nn.Linear(out_dim, out_dim >> 1),
            nn.ReLU(),
            nn.Linear(out_dim >> 1, out_dim >> 2),
            nn.ReLU(),
            nn.Linear(out_dim >> 2, out_dim >> 3),
            nn.ReLU(),
            nn.Linear(out_dim >> 3, n_classes),
        )

    def forward(self, x, edge_index, batch):
        h = self.embedding_h(x)

        h = self.conv(h, edge_index)

        h = self.MLP_layer(h)

        h = global_mean_pool(h, batch)
        return h

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        return criterion(pred, label)


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_classes):
        super().__init__()

        self.conv = Sequential(
            "x, edge_index",
            [
                (GCNConv(in_dim, hidden_dim), "x, edge_index -> x"),
                nn.ReLU(inplace=True),
                (GCNConv(hidden_dim, hidden_dim), "x, edge_index -> x"),
                nn.ReLU(inplace=True),
                (GCNConv(hidden_dim, hidden_dim), "x, edge_index -> x"),
                nn.ReLU(inplace=True),
                (GCNConv(hidden_dim, n_classes), "x, edge_index -> x"),
                # nn.Softmax(dim=1),
            ],
        )

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)

        x = global_mean_pool(x, batch)
        return x

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        return criterion(pred, label)
