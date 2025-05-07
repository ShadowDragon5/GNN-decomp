import torch
import torch.nn as nn
from torch_geometric.nn import Sequential

from .common import ResidualGCNLayer


class GCN_CN(nn.Module):
    """
    Adapted from:
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/SBMs_node_classification/gcn_net.py
    """

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
        # weighted cross entropy for unbalanced classes
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(label.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        criterion = nn.CrossEntropyLoss(weight=weight)
        return criterion(pred, label)
