import re

import torch
from sklearn.cluster import spectral_clustering
from torch_geometric.data import Data
from torch_geometric.index import index2ptr
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import sort_edge_index


class PartitionedData(Data):
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


def partition_transform(data: Data, num_parts: int = 2):
    """Spectral graph decomposition"""
    data = position_transform(data)

    assert data.x is not None
    assert data.y is not None
    assert data.edge_index is not None

    N = data.x.shape[0]
    A = torch.zeros((N, N), dtype=torch.float)  # adjacency matrix

    # bidirectional adjacency matrix
    A[data.edge_index[0], data.edge_index[1]] = 1
    A[data.edge_index[1], data.edge_index[0]] = 1

    # # Laplacian
    # L = torch.diag(torch.sum(A, dim=-1)) - A
    #
    # # eigenvalue decomposition
    # with torch.no_grad():
    #     # eigenvalues, eigenvectors
    #     (_, Q) = torch.linalg.eigh(L)
    #     x2 = Q[-2]
    #
    #     partitions = []
    #     # TODO: more partitions with a clustering algorithm
    #     partitions.append((x2 >= 0).int())
    #     partitions.append((x2 < 0).int())
    #
    #     # # graph partitioning
    #     # parts = torch.Tensor(
    #     #     [part_to_data(data.x[p], data.y, A[p][:, p]) for p in partitions]
    #     # )
    #
    #     data.parts = torch.stack(partitions)

    labels = spectral_clustering(A, n_clusters=num_parts)
    subgraphs = dict()
    for i in range(num_parts):
        G = data.subgraph(torch.tensor(labels == i))
        subgraphs[f"x_{i}"] = G.x
        subgraphs[f"edge_index_{i}"] = G.edge_index

    return PartitionedData(y=data.y, batch=data.batch, **subgraphs)


def partition_transform_METIS(data: Data, num_parts: int):
    """Concatenates features and position, then partitions the graph using METIS."""
    x = torch.cat((torch.Tensor(data.x), torch.Tensor(data.pos)), dim=1)

    assert data.edge_index is not None, "Edge index must be provided"

    print(data.num_nodes, data.edge_index.shape)
    # Convert to CSR format for METIS
    # row, index = sort_edge_index(data.edge_index, num_nodes=data.num_nodes)
    # indptr = index2ptr(row, size=data.num_nodes)

    index, col = sort_edge_index(
        data.edge_index, num_nodes=data.num_nodes, sort_by_row=False
    )
    indptr = index2ptr(col, size=data.num_nodes)

    print(indptr.shape, index.shape)
    # Compute METIS partitioning
    cluster = pyg_lib.partition.metis(indptr.cpu(), index.cpu(), num_parts)

    # subgraphs = []
    # for part_id in range(num_parts):
    #     partition_nodes = torch.where(cluster == part_id)[0]
    #     # Create a node mapping
    #     node_map = {n: i for i, n in enumerate(partition_nodes)}
    #
    #     # # Filter and remap edges
    #     # edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
    #     # sub_edge_index = data.edge_index[:, edge_mask]
    #     # sub_edge_index = torch.tensor([[node_map[n.item()] for n in edge] for edge in sub_edge_index], dtype=torch.long).T
    #
    #     # Filter and remap edges
    #     edge_mask = torch.tensor(
    #         [n in node_map and m in node_map for n, m in zip(*data.edge_index)]
    #     )
    #     sub_edge_index = data.edge_index[:, edge_mask]
    #     sub_edge_index = torch.tensor(
    #         [[node_map[n] for n in edge] for edge in sub_edge_index]
    #     )
    #
    #     # Create a subgraph Data object
    #     subgraph = Data(
    #         x=x[partition_nodes],
    #         y=data.y,
    #         edge_index=sub_edge_index,
    #         num_nodes=len(partition_nodes),
    #     )
    #     subgraphs.append(subgraph)

    return Data(
        x=x,
        y=data.y,
        edge_index=data.edge_index,
        batch=data.batch,
    )


# def partition_transform(data: Data, partitions: int):
#     x = torch.cat((torch.Tensor(data.x), torch.Tensor(data.pos)), 1)
#
#     cluster_data = ClusterData(
#         Data(
#             x=x,
#             y=data.y,
#             edge_index=data.edge_index,
#             batch=data.batch,
#         ),
#         num_parts=partitions,
#     )
#     out = [data for data in cluster_data]
#     return out


# def write_results(
#     filename: str,
#     epoch=None,
#     train_loss=None,
#     valid_loss=None,
#     valid_acc=None,
#     test_acc=None,
# ):
#     if train_loss and valid_loss and valid_acc:
#         mlflow.log_metrics(
#             {
#                 "train/loss": train_loss,
#                 "validate/loss": valid_loss,
#                 "validate/accuracy": valid_acc,
#             },
#             step=epoch,
#         )
#     if test_acc:
#         mlflow.log_metric("test/accuracy", test_acc)
#
#     with open(f"results/{filename}", "a") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([epoch, train_loss, valid_loss, valid_acc, test_acc])
