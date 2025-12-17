"""
Implementation of full Graclus clustering algorithm based on a paper
Weighted Graph Cuts without Eigenvectors: A Multilevel Approach
"""

import math

import torch
from sklearn.cluster import spectral_clustering
from torch_geometric.data import Data
from torch_geometric.nn import graclus, max_pool
from torch_geometric.utils import to_scipy_sparse_matrix


def graclus_refine(edge_index, clusters, num_clusters, iters=5, boundary_only=True):
    row, col = edge_index
    N = clusters.size(0)
    device = clusters.device

    # adjacency weights assumed 1
    w = torch.ones(row.size(0), device=device)

    for _ in range(iters):
        vol = torch.zeros(num_clusters, device=device)
        vol.scatter_add_(0, clusters[row], w)

        same = clusters[row] == clusters[col]
        internal = torch.zeros(num_clusters, device=device)
        internal.scatter_add_(0, clusters[row][same], w[same])

        if boundary_only:
            boundary = clusters[row] != clusters[col]
            nodes = torch.unique(torch.cat([row[boundary], col[boundary]]))
        else:
            nodes = torch.arange(N, device=device)

        changed = False

        for u in nodes:
            mask = row == u
            nbrs = col[mask]
            w_uv = w[mask]

            link_uc = torch.zeros(num_clusters, device=device)
            link_uc.scatter_add_(0, clusters[nbrs], w_uv)

            # kernel k-means score
            score = 2 * link_uc / (vol + 1e-12) - internal / (vol + 1e-12) ** 2

            new_c = score.argmax()
            if new_c != clusters[u]:
                clusters[u] = new_c
                changed = True

        if not changed:
            break

    return clusters


def graclus_kway(data: Data, parts: int, refine_iters=5):
    assert data.num_nodes is not None and data.num_nodes > 0

    graphs = [data]
    cluster_maps = []

    # heuristic from the paper
    pre_parts = 5 * parts
    num_levels = (
        math.floor(math.log2(data.num_nodes)) - math.ceil(math.log2(pre_parts)) + 1
    )

    # 1. Coarsen
    for _ in range(num_levels):
        cluster = graclus(data.edge_index, num_nodes=data.num_nodes)  # type: ignore

        # reindex cluster ids
        _, cluster = torch.unique(cluster, return_inverse=True)

        cluster_maps.append(cluster)
        data = max_pool(cluster, data)
        graphs.append(data)

    # 2. Cluster
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)  # type: ignore
    labels = spectral_clustering(A, n_clusters=parts)
    clusters = torch.tensor(labels, device=data.edge_index.device)  # type: ignore

    # 3. Uncoarsen and refine
    for level in reversed(range(num_levels)):
        # project clustering down
        clusters = clusters[cluster_maps[level]]

        # refine on this level
        clusters = graclus_refine(
            edge_index=graphs[level].edge_index,
            clusters=clusters,
            num_clusters=parts,
            iters=refine_iters,
            boundary_only=True,
        )

    return clusters
