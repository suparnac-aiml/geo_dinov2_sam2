"""
Tests multiple threshold percentile values on a fixed subset of pairs
to find which threshold setting maximizes IoU, before committing to
a full 348-pair run with the chosen value.
"""

import sys
sys.path.append('.')
import json
import numpy as np

from dataset import load_levircd, get_pair
from feature_extractor import DINOv2FeatureExtractor
from comparison import compute_change_heatmap
from thresholding import upsample_to_image_size
from sam2_refiner import SAM2Refiner
from evaluation import prepare_ground_truth, compute_metrics

# Test these threshold percentiles
CANDIDATE_THRESHOLDS = [70, 75, 80, 85, 90, 95]

# Use a fixed subset for speed — same 30 pairs across all threshold tests
# so comparisons are fair (apples to apples)
SUBSET_SIZE = 30


def threshold_at_percentile(heatmap, percentile):
    """Same logic as thresholding.py but takes percentile as a parameter
    instead of reading from config — needed so we can sweep multiple values."""
    flat_values = heatmap.flatten().numpy()
    threshold_value = np.percentile(flat_values, percentile)
    binary_mask = (heatmap > threshold_value).float()
    return binary_mask


def main():
    dataset = load_levircd()
    subset_indices = list(range(SUBSET_SIZE))  # first 30 pairs, fixed

    print("Loading models...")
    extractor = DINOv2FeatureExtractor()
    refiner = SAM2Refiner()

    # Pre-compute DINOv2 features once per pair — reused across all
    # threshold values, since DINOv2 extraction doesn't change with
    # threshold. Only the thresholding + SAM2 step changes.
    print("Pre-computing DINOv2 features for subset...")
    precomputed = []
    for idx in subset_indices:
        t1, t2, gt_mask_pil = get_pair(dataset, idx)
        feat_t1, grid_size = extractor.extract_patch_features(t1)
        feat_t2, _ = extractor.extract_patch_features(t2)
        heatmap = compute_change_heatmap(feat_t1, feat_t2, grid_size)
        gt_mask = prepare_ground_truth(gt_mask_pil)
        precomputed.append({
            "t2": np.array(t2),
            "heatmap": heatmap,
            "gt_mask": gt_mask,
            "image_size": t1.size[::-1]
        })
    print("Done precomputing.\n")

    results = {}

    for pct in CANDIDATE_THRESHOLDS:
        print(f"Testing threshold percentile: {pct}")
        ious = []

        for item in precomputed:
            binary_mask = threshold_at_percentile(item["heatmap"], pct)
            upsampled = upsample_to_image_size(binary_mask, item["image_size"])
            final_mask = refiner.refine_mask(item["t2"], upsampled.numpy())
            metrics = compute_metrics(final_mask, item["gt_mask"])
            ious.append(metrics["iou"])

        avg_iou = np.mean(ious)
        results[pct] = avg_iou
        print(f"  -> Average IoU on subset: {avg_iou:.4f}\n")

    print("=" * 50)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 50)
    for pct, iou in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  Percentile {pct}: IoU = {iou:.4f}")

    best_pct = max(results, key=results.get)
    print(f"\nBest threshold: {best_pct} (IoU = {results[best_pct]:.4f})")

    with open("/home/ubuntu/geo_dinov2_sam2/results/threshold_sweep.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
