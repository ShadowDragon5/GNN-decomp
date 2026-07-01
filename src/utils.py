import re

import torch
from sklearn.cluster import spectral_clustering
from torch_geometric.data import Data
from torch_geometric.nn import graclus, radius_graph
from torch_geometric.nn.pool import avg_pool
from torch_geometric.nn.pool.avg_pool import _avg_pool_x
from torch_geometric.utils import to_scipy_sparse_matrix

from graclus import graclus_kway


class PartitionedData(Data):
    def get(self, attr: str, i: int, device) -> torch.Tensor | None:
        if attr == "batch":
            return self.get_batch(i, device)

        a = getattr(self, f"{attr}_{i}", None)
        if a is None:
            return None
        return a.to(device)

    def set(self, attr: str, i: int, value) -> None:
        setattr(self, f"{attr}_{i}", value)

    def get_batch(self, i: int, device: torch.device) -> torch.Tensor | None:
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


torch.serialization.add_safe_globals([PartitionedData])


def get_data(
    data: Data | PartitionedData,
    i: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    data: Data object from which the `keys` will be extracted into a dictionary
    i: (optional) partition index of the partitioned data
    device: (optional) is only needed if i is provided and sets the device of the parameter tensor
    """
    keys = [
        "x",
        "x_eval",
        "y",
        "edge_index",
        "edge_attr",
        "batch",
        "v_gt",
        "gt",
    ]

    def wrapped_get(k: str) -> torch.Tensor | None:
        if i is None:
            return getattr(data, k, None)
        return data.get(k, i, device)

    if wrapped_get("edge_index") is not None:
        return {k: wrapped_get(k) for k in keys}

    # Sample and add `edge_index` to `AirfRANS` data
    pos = wrapped_get("pos")
    assert pos is not None

    # sample points
    n = pos.size(0)
    sampleN = 32000

    if isinstance(data, PartitionedData):
        for j in range(6):
            if data.get("x", j, device) is None:
                break
        sampleN //= j  # type: ignore

    if n <= sampleN:
        idx = torch.arange(n, device=device)
    else:
        idx = torch.multinomial(torch.ones(n, device=device), sampleN)

    edge_index = radius_graph(
        x=pos[idx],
        r=0.05,
        loop=True,
        max_num_neighbors=64,
    )

    return {
        "edge_index": edge_index,
        **{
            k: wrapped_get(k)[idx]  # type: ignore
            for k in ["x", "y"]
        },
    }


def coarsen_graph_avg(data: Data, level=1) -> Data:
    data.to("cpu")  # HACK: graclus hangs on GPU
    for _ in range(level):
        if data.edge_index is None:
            data = Data(**get_data(data))
        cluster = graclus(data.edge_index, num_nodes=data.num_nodes)  # type: ignore
        # reindex cluster ids
        _, cluster = torch.unique(cluster, return_inverse=True)

        # pool labels if they are per node
        assert isinstance(data.y, torch.Tensor)
        y = (
            data.y
            if data.y.shape[0] != data.x.shape[0]  # type: ignore
            else _avg_pool_x(cluster, data.y)
        )
        data = avg_pool(cluster, data)
        data.y = y

    return data


def coarsen_graph_airfrans(data: Data, level=1, radius=0.05) -> Data:
    assert data.num_nodes is not None
    assert data.pos is not None
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.y, torch.Tensor)

    n = data.num_nodes // (2**level)
    idx = torch.multinomial(torch.ones(data.num_nodes), n)

    edge_index = radius_graph(
        x=data.pos[idx],
        r=radius,
        loop=True,
        max_num_neighbors=64,
    )

    data.num_nodes = n
    data.edge_index = edge_index
    data.x = data.x[idx]
    data.y = data.y[idx]

    return data


def position_transform(data: Data) -> Data:
    """Concatenates features and position"""
    x = torch.cat((torch.Tensor(data.x), torch.Tensor(data.pos)), 1)
    return Data(
        x=x,
        y=data.y,
        pos=data.pos,
        edge_index=data.edge_index,
        batch=data.batch,
    )


def normalization_transform(data: Data, mean_x, std_x, mean_y, std_y) -> Data:
    x = data.x
    y = data.y
    x = (x - mean_x) / (std_x + 1e-8)
    y = (y - mean_y) / (std_y + 1e-8)

    return Data(
        x=x,
        y=y,
        pos=data.pos,
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
    assert data.edge_index is not None

    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.x.shape[0])

    labels = spectral_clustering(A, n_clusters=num_parts)
    subgraphs = dict()

    for i in range(num_parts):
        G = data.subgraph(torch.tensor(labels == i))
        subgraphs[f"x_{i}"] = G.x
        subgraphs[f"edge_index_{i}"] = G.edge_index
        subgraphs[f"y_{i}"] = getattr(G, "y", None)
        subgraphs[f"edge_attr_{i}"] = getattr(G, "edge_attr", None)
        subgraphs[f"current_u_{i}"] = getattr(G, "current_u", None)
        subgraphs[f"h_{i}"] = getattr(G, "h", None)
        subgraphs[f"gt_{i}"] = getattr(G, "gt", None)
        subgraphs[f"v_gt_{i}"] = getattr(G, "v_gt", None)
        subgraphs[f"x_eval_{i}"] = getattr(G, "x_eval", None)
        subgraphs[f"coords_{i}"] = getattr(G, "coords", None)
        subgraphs[f"unroll_v_gt_{i}"] = getattr(G, "unroll_v_gt", None)
        subgraphs[f"unroll_u_gt_{i}"] = getattr(G, "unroll_u_gt", None)
        subgraphs[f"a_gt_{i}"] = getattr(G, "a_gt", None)

    return PartitionedData(batch=data.batch, **subgraphs)


def partition_data_points_graclus(data: Data, num_parts: int = 2):
    assert data.x is not None
    assert data.pos is not None

    edge_index = radius_graph(
        x=data.pos,
        r=0.05,
        loop=True,
        max_num_neighbors=64,
    )
    data.edge_index = edge_index

    clusters = graclus_kway(data, num_parts)

    subgraphs = dict()
    for i in range(num_parts):
        G = data.subgraph(clusters == i)
        subgraphs[f"x_{i}"] = G.x
        subgraphs[f"pos_{i}"] = G.pos
        subgraphs[f"y_{i}"] = G.y

    return PartitionedData(batch=data.batch, **subgraphs)


def morton_partition(pos, num_parts):
    N = pos.size(0)

    mins = pos.min(0).values
    maxs = pos.max(0).values

    x = ((pos[:, 0] - mins[0]) / (maxs[0] - mins[0] + 1e-12) * 1024).long()
    y = ((pos[:, 1] - mins[1]) / (maxs[1] - mins[1] + 1e-12) * 1024).long()

    code = torch.zeros_like(x)
    for i in range(10):
        code |= ((x >> i) & 1) << (2 * i)
        code |= ((y >> i) & 1) << (2 * i + 1)

    perm = torch.argsort(code)

    # balanced slicing
    splits = torch.linspace(0, N, num_parts + 1).long()

    labels = torch.empty(N, dtype=torch.long, device=pos.device)

    for i in range(num_parts):
        labels[perm[splits[i] : splits[i + 1]]] = i

    return labels


def partition_data_points_morton(data: Data, num_parts: int = 2):
    assert data.x is not None
    assert data.pos is not None

    clusters = morton_partition(data.pos, num_parts)

    subgraphs = dict()
    for i in range(num_parts):
        G = data.subgraph(clusters == i)
        subgraphs[f"x_{i}"] = G.x
        subgraphs[f"pos_{i}"] = G.pos
        subgraphs[f"y_{i}"] = G.y

    return PartitionedData(batch=data.batch, **subgraphs)


def init_weights(m: torch.nn.Module):
    """Performs weight initialization.

    Args:
        m: PyTorch module

    """
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, torch.nn.Linear):
        m.weight.data = torch.nn.init.xavier_uniform_(
            m.weight.data, gain=torch.nn.init.calculate_gain("relu")
        )
        if m.bias is not None:
            m.bias.data.zero_()
