"""
Tests whether IoU correlates with the size of the ground-truth
changed region, to confirm or refute the hypothesis that our
method performs better on large changes and worse on small ones.
"""

import json
import numpy as np
from datasets import load_dataset
from evaluation import prepare_ground_truth

with open("/home/ubuntu/geo_dinov2_sam2/results/metrics_v2_full.json", "r") as f:
    all_metrics = json.load(f)

print("Loading dataset to compute ground truth sizes...")
ds = load_dataset("blanchon/LEVIR_CDPlus", split="test")

gt_sizes = []
ious = []

for m in all_metrics:
    idx = m["pair_index"]
    gt_mask_pil = ds[idx]["mask"]
    gt_mask = prepare_ground_truth(gt_mask_pil)
    gt_sizes.append(gt_mask.sum())
    ious.append(m["iou"])

gt_sizes = np.array(gt_sizes)
ious = np.array(ious)

correlation = np.corrcoef(gt_sizes, ious)[0, 1]
print(f"\nCorrelation between ground-truth change size and IoU: {correlation:.4f}")

# Split into small vs large change pairs by median size
median_size = np.median(gt_sizes)
small_mask = gt_sizes <= median_size
large_mask = gt_sizes > median_size

print(f"\nMedian GT change size: {median_size:.0f} pixels")
print(f"Mean IoU on SMALL-change pairs (below median): {ious[small_mask].mean():.4f}")
print(f"Mean IoU on LARGE-change pairs (above median): {ious[large_mask].mean():.4f}")
