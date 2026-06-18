"""
Computes standard change-detection evaluation metrics by comparing
our predicted mask against the LEVIR-CD+ ground truth mask.
"""

import numpy as np

def prepare_ground_truth(mask_pil):
    """
    Converts the ground truth PIL mask into a clean binary numpy array.

    Args:
        mask_pil: PIL Image, mode 'L' (grayscale)

    Returns:
        numpy array (H, W), values 0 or 1
    """
    mask_np = np.array(mask_pil)
    # This dataset mirror stores masks as 0/1 directly (not 0/255).
    # We threshold at 0 (>0 means "any nonzero value counts as changed")
    # rather than assuming a specific max value, making this robust
    # to either 0/1 or 0/255 encodings.
    binary_gt = (mask_np > 0).astype(np.uint8)
    return binary_gt


def compute_metrics(predicted_mask, ground_truth_mask):
    """
    Args:
        predicted_mask: numpy array (H, W), values 0 or 1 — our output
        ground_truth_mask: numpy array (H, W), values 0 or 1 — true labels

    Returns:
        dict with keys: iou, precision, recall, f1, dice
    """
    pred = predicted_mask.astype(bool)
    gt = ground_truth_mask.astype(bool)

    # True Positive: both predicted AND actually changed
    tp = np.logical_and(pred, gt).sum()
    # False Positive: we said changed, but it wasn't
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    # False Negative: we missed a real change
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    # Avoid division by zero if a pair has zero predicted or zero true change
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    dice = f1  # Dice coefficient equals F1 for binary masks — same formula

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice)
    }
