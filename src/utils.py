import re

import torch
from sklearn.cluster import spectral_clustering
from torch_geometric.data import Data
from torch_geometric.nn import graclus, radius_graph
from torch_geometric.utils import to_scipy_sparse_matrix


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


def partition_data_points(data: Data, num_parts: int = 2):
    assert data.x is not None
    assert data.pos is not None

    N = data.pos.size(0)

    edge_index = radius_graph(
        x=data.pos,
        r=0.05,
        loop=True,
        max_num_neighbors=64,
    )
    data.edge_index = edge_index

    clusters = torch.zeros(N, dtype=torch.long)

    sizes = {0: N}
    for p in range(1, num_parts):
        idx: int = max(sizes, key=sizes.get)  # type: ignore

        mask = clusters == idx
        global_nodes = mask.nonzero(as_tuple=False).view(-1)

        Gp = data.subgraph(mask)
        assert Gp.edge_index is not None

        cl = graclus(Gp.edge_index, num_nodes=Gp.num_nodes) % 2

        clusters[global_nodes[cl == 0]] = idx
        clusters[global_nodes[cl == 1]] = p

        sizes[idx] = (cl == 0).sum().item()  # type: ignore
        sizes[p] = (cl == 1).sum().item()  # type: ignore

    subgraphs = dict()
    for i in range(num_parts):
        G = data.subgraph(clusters == i)
        subgraphs[f"x_{i}"] = G.x
        subgraphs[f"pos_{i}"] = G.pos
        subgraphs[f"y_{i}"] = G.y

    return PartitionedData(batch=data.batch, **subgraphs)
