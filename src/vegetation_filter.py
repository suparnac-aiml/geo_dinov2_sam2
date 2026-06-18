"""
Heuristic vegetation detection from RGB-only satellite imagery.

True NDVI (Normalized Difference Vegetation Index) requires a
near-infrared band, which we don't have in this RGB-only dataset.
Instead, we use a simplified RGB-based greenness heuristic as an
approximation — this is a known limitation we document honestly.
"""

import numpy as np


def compute_greenness_mask(image_np, threshold=1.15):
    """
    Args:
        image_np: numpy array (H, W, 3), RGB image, values 0-255
        threshold: ratio above which a pixel is considered vegetation

    Returns:
        vegetation_mask: numpy array (H, W), values 0 or 1
                          1 = likely vegetation, 0 = likely not
    """
    r = image_np[:, :, 0].astype(np.float32)
    g = image_np[:, :, 1].astype(np.float32)
    b = image_np[:, :, 2].astype(np.float32)

    epsilon = 1e-7
    # Simple greenness ratio: vegetation reflects more green than red/blue
    greenness_ratio = g / ((r + b) / 2 + epsilon)

    vegetation_mask = (greenness_ratio > threshold).astype(np.uint8)
    return vegetation_mask


def suppress_vegetation_false_positives(predicted_mask, t1_image_np, t2_image_np):
    """
    Removes predicted change regions where BOTH T1 and T2 are
    classified as vegetation — i.e., "this was trees before AND
    after, so it's probably not a real change, just seasonal/
    visual noise being misread as change."

    Args:
        predicted_mask: numpy array (H, W), our SAM2 output, 0/1
        t1_image_np: numpy array (H, W, 3), T1 RGB image
        t2_image_np: numpy array (H, W, 3), T2 RGB image

    Returns:
        filtered_mask: numpy array (H, W), 0/1, with vegetation-only
                        regions suppressed
    """
    veg_t1 = compute_greenness_mask(t1_image_np)
    veg_t2 = compute_greenness_mask(t2_image_np)

    # "Persistent vegetation" = green in both images = likely not real change
    persistent_vegetation = np.logical_and(veg_t1, veg_t2)

    # Suppress predicted change wherever persistent vegetation is detected
    filtered_mask = np.logical_and(
        predicted_mask, np.logical_not(persistent_vegetation)
    ).astype(np.uint8)

    return filtered_mask
