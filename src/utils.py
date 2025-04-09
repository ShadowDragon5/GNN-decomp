import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.cluster import spectral_clustering
from torch_geometric.data import Data


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


def partition_transform_global(data: Data, num_parts: int = 2):
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

    labels = spectral_clustering(A, n_clusters=num_parts)
    subgraphs = dict()
    for i in range(num_parts):
        G = data.subgraph(torch.tensor(labels == i))
        subgraphs[f"x_{i}"] = G.x
        subgraphs[f"edge_index_{i}"] = G.edge_index

    return PartitionedData(y=data.y, batch=data.batch, **subgraphs)


# def partition_transform(data: Data, num_parts: int = 2):
#     """Spectral graph decomposition with plotting"""
#     data = position_transform(data)
#
#     assert data.x is not None
#     assert data.y is not None
#     assert data.edge_index is not None
#
#     N = data.x.shape[0]
#     print(data.x.shape)
#     A = torch.zeros((N, N), dtype=torch.float)  # adjacency matrix
#
#     # bidirectional adjacency matrix
#     A[data.edge_index[0], data.edge_index[1]] = 1
#     A[data.edge_index[1], data.edge_index[0]] = 1
#
#     labels = spectral_clustering(A, n_clusters=num_parts)
#     subgraphs = dict()
#
#     # Create a NetworkX graph for plotting
#     G = nx.Graph()
#     for i in range(N):
#         # Use the RGB values from the first 3 columns of data.x for color
#         node_color = data.x[i][
#             :3
#         ].numpy()  # RGB values are the first 3 columns of data.x
#         node_pos = data.x[i][3:].numpy()
#         G.add_node(i, pos=node_pos, color=node_color)
#
#     # Assigning colors to clusters
#     cluster_colors = plt.cm.get_cmap("tab20", num_parts)  # Get a color map for clusters
#
#     for i, (node1, node2) in enumerate(data.edge_index.t()):
#         G.add_edge(node1.item(), node2.item())
#
#     # Plot the graph with node colors and cluster outlines
#     plt.figure(figsize=(8, 8))
#     ax = plt.gca()
#
#     print(data.x.shape)
#     print(data.x[data.x.shape[0] - 1])
#     # Draw nodes with colors from their RGB values
#     for i in range(num_parts):
#         cluster_nodes = [node for node, attr in G.nodes(data=True) if labels[node] == i]
#         cluster_color = cluster_colors(i)  # Color for this cluster
#
#         # Draw nodes of this cluster with the correct color
#         nx.draw_networkx_nodes(
#             G,
#             data.x[:][3:],
#             nodelist=cluster_nodes,
#             node_color=[G.nodes[node]["color"] for node in cluster_nodes],
#             node_size=100,
#             edgecolors=[cluster_color] * len(cluster_nodes),
#             linewidths=2,
#             ax=ax,
#         )
#
#     # Draw edges
#     nx.draw_networkx_edges(G, data.x[:][3:], alpha=0.5, ax=ax)
#
#     # Optional: Draw node labels if necessary (you can skip this step for large graphs)
#     # nx.draw_networkx_labels(G, data.pos, ax=ax)
#
#     plt.title(f"Graph with {num_parts} clusters")
#     plt.axis("off")
#     plt.show()
#
#     # Return partitioned data as before
#     subgraphs = dict()
#     for i in range(num_parts):
#         G_sub = data.subgraph(torch.tensor(labels == i))
#         subgraphs[f"x_{i}"] = G_sub.x
#         subgraphs[f"edge_index_{i}"] = G_sub.edge_index
#
#     return PartitionedData(y=data.y, batch=data.batch, **subgraphs)
