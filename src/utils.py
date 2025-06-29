import re

import torch
from sklearn.cluster import spectral_clustering
from torch_geometric.data import Data


class PartitionedData(Data):
    def get_x(self, i, device):
        return getattr(self, f"x_{i}").to(device)

    def get_y(self, i, device):
        return getattr(self, f"y_{i}").to(device)

    def get_edge_index(self, i, device):
        return getattr(self, f"edge_index_{i}").to(device)

    def get_batch(self, i, device):
        # if (batch := getattr(self, f"batch_{i}", None)) is not None:
        if (batch := getattr(self, f"x_{i}_batch", None)) is not None:
            return batch.to(device)
        # datasets like PATTERN don't have batch masks
        return None

    def __inc__(self, key, value, *args, **kwargs):
        if m := re.match(r"edge_index_(\d+)", key):
            x = getattr(self, f"x_{m.group(1)}")
            return x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def position_transform(data: Data) -> Data:
    """Concatenates features and position"""
    x = torch.cat((torch.Tensor(data.x), torch.Tensor(data.pos)), 1)
    return Data(
        x=x,
        y=data.y,
        edge_index=data.edge_index,
        batch=data.batch,
    )


def part_to_data(x, y, A) -> Data:
    adj = torch.transpose(A, -2, -1)
    index = adj.nonzero(as_tuple=True)
    return Data(x=x, y=y, edge_index=torch.stack(index, 0))


def partition_transform_global(data: Data, num_parts: int = 2):
    """Spectral graph decomposition"""
    assert data.x is not None
    assert data.y is not None
    assert data.edge_index is not None

    N = data.x.shape[0]
    A = torch.zeros((N, N), dtype=torch.float)  # adjacency matrix

    # bidirectional adjacency matrix
    A[data.edge_index[0], data.edge_index[1]] = 1
    A[data.edge_index[1], data.edge_index[0]] = 1

    labels = spectral_clustering(A, n_clusters=num_parts)
    subgraphs = dict()
    for i in range(num_parts):
        G = data.subgraph(torch.tensor(labels == i))
        subgraphs[f"x_{i}"] = G.x
        subgraphs[f"y_{i}"] = G.y
        subgraphs[f"edge_index_{i}"] = G.edge_index

    return PartitionedData(batch=data.batch, **subgraphs)
