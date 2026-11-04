"""Anomalous points -> target clusters (paper Sec. III-B, "Explore the unseen").

Points above the threshold are linked whenever they lie within `radius` of
each other (ball query) and the connected components are the clusters. A
cluster too wide to fit in one view from `d_max` away is cut in two with
k-means, repeatedly, until every piece fits - the paper does not split, but
without it a long wall rim asks for a standoff larger than the room.
"""

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def ball_query_clusters(xyz, mask, radius=0.15, min_pts=30):
    """-> list of row-index arrays into xyz, largest cluster first."""
    idx = np.flatnonzero(mask)
    if len(idx) < min_pts:
        return []
    pairs = cKDTree(xyz[idx]).query_pairs(radius, output_type="ndarray")
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                     shape=(len(idx), len(idx)))
    _, label = connected_components(adj, directed=False)
    sizes = np.bincount(label)
    keep = np.flatnonzero(sizes >= min_pts)
    keep = keep[np.argsort(-sizes[keep])]
    return [idx[label == k] for k in keep]


def cluster_radius(pts):
    """r_k = max distance from the anchor (centroid) to a cluster point."""
    return float(np.linalg.norm(pts - pts.mean(0), axis=1).max())


def split_wide(xyz, clusters, max_radius, min_pts=30, seed=0):
    """Bisect any cluster whose radius exceeds max_radius until none does."""
    out, todo = [], list(clusters)
    while todo:
        c = todo.pop()
        if len(c) < min_pts:
            continue
        if cluster_radius(xyz[c]) <= max_radius or len(c) < 2 * min_pts:
            out.append(c)
            continue
        _, lab = kmeans2(xyz[c].astype(np.float64), 2, minit="++", seed=seed)
        if lab.min() == lab.max():                 # degenerate split, keep as is
            out.append(c)
            continue
        todo.extend([c[lab == 0], c[lab == 1]])
    return sorted(out, key=len, reverse=True)


def find_clusters(xyz, score, tau=0.25, radius=0.15, min_pts=30, max_radius=None):
    clusters = ball_query_clusters(xyz, score > tau, radius, min_pts)
    if max_radius is not None:
        clusters = split_wide(xyz, clusters, max_radius, min_pts)
    return clusters
