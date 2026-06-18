"""
Compares DINOv2 feature embeddings between T1 and T2 to produce
a coarse change heatmap at patch-grid resolution.
"""

import torch
import torch.nn.functional as F


def compute_change_heatmap(features_t1, features_t2, grid_size):
    """
    Args:
        features_t1: tensor (num_patches, 768) — T1 embeddings
        features_t2: tensor (num_patches, 768) — T2 embeddings
        grid_size: (height_patches, width_patches) tuple

    Returns:
        heatmap: 2D tensor (grid_h, grid_w) where higher values = more change
                 values are in range [0, 2] (1 - cosine_similarity)
    """
    # Cosine similarity per patch: how aligned are the two 768-dim vectors?
    # dim=1 means we compare each patch's full 768-dim vector independently
    similarity = F.cosine_similarity(features_t1, features_t2, dim=1)

    # similarity ranges from -1 (opposite) to 1 (identical)
    # we convert to a "distance" so higher number = more change
    distance = 1 - similarity  # now ranges from 0 (no change) to 2 (max change)

    grid_h, grid_w = grid_size
    heatmap = distance.reshape(grid_h, grid_w)

    return heatmap
