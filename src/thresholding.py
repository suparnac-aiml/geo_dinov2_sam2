"""
Converts a continuous change heatmap into discrete candidate regions
by thresholding, then upsamples to full image resolution.
"""

import torch
import numpy as np
from config import CHANGE_THRESHOLD_PERCENTILE


def threshold_heatmap(heatmap):
    """
    Args:
        heatmap: 2D tensor (grid_h, grid_w) of change scores

    Returns:
        binary_mask: 2D tensor (grid_h, grid_w), 1 = candidate change, 0 = unchanged
        threshold_value: the actual numeric threshold used (for logging/debugging)
    """
    # Flatten to compute percentile across all patches
    flat_values = heatmap.flatten().numpy()
    threshold_value = np.percentile(flat_values, CHANGE_THRESHOLD_PERCENTILE)

    binary_mask = (heatmap > threshold_value).float()

    return binary_mask, threshold_value


def upsample_to_image_size(grid_mask, target_size):
    """
    Args:
        grid_mask: 2D tensor (grid_h, grid_w), low-resolution mask
        target_size: (height, width) of the original image, e.g. (1024, 1024)

    Returns:
        upsampled: 2D tensor (target_h, target_w), same values, full resolution
    """
    import torch.nn.functional as F

    # F.interpolate expects shape (batch, channels, height, width)
    grid_mask_4d = grid_mask.unsqueeze(0).unsqueeze(0)

    upsampled = F.interpolate(
        grid_mask_4d,
        size=target_size,
        mode='nearest'  # nearest-neighbor: no smoothing/blending of mask values
    )

    return upsampled.squeeze(0).squeeze(0)
