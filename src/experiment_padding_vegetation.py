"""
Tests box padding and vegetation filtering — separately and combined —
on the same fixed 30-pair subset, using DINOv2-base + threshold 85
(our established best baseline), to see which improvements actually help.
"""

import sys
sys.path.append('.')
import numpy as np

from dataset import load_levircd, get_pair
from feature_extractor import DINOv2FeatureExtractor
from comparison import compute_change_heatmap
from thresholding import upsample_to_image_size
from sam2_refiner import SAM2Refiner
from evaluation import prepare_ground_truth, compute_metrics
from vegetation_filter import suppress_vegetation_false_positives

SUBSET_SIZE = 30
BEST_THRESHOLD_PCT = 85
PADDING_PIXELS = 15  # roughly one DINOv2 patch-width of extra context


def threshold_at_percentile(heatmap, percentile):
    flat_values = heatmap.flatten().numpy()
    threshold_value = np.percentile(flat_values, percentile)
    return (heatmap > threshold_value).float()


def main():
    dataset = load_levircd()
    subset_indices = list(range(SUBSET_SIZE))

    print("Loading models...")
    extractor = DINOv2FeatureExtractor()
    refiner = SAM2Refiner()

    print("Pre-computing DINOv2 features + base masks for subset...")
    precomputed = []
    for idx in subset_indices:
        t1, t2, gt_mask_pil = get_pair(dataset, idx)
        feat_t1, grid_size = extractor.extract_patch_features(t1)
        feat_t2, _ = extractor.extract_patch_features(t2)
        heatmap = compute_change_heatmap(feat_t1, feat_t2, grid_size)
        binary_mask = threshold_at_percentile(heatmap, BEST_THRESHOLD_PCT)
        upsampled = upsample_to_image_size(binary_mask, t1.size[::-1])
        gt_mask = prepare_ground_truth(gt_mask_pil)

        precomputed.append({
            "t1_np": np.array(t1),
            "t2_np": np.array(t2),
            "candidate_np": upsampled.numpy(),
            "gt_mask": gt_mask
        })
    print("Done.\n")

    configs = {
        "baseline (no padding, no veg filter)": {"padding": 0, "veg_filter": False},
        "padding only":                          {"padding": PADDING_PIXELS, "veg_filter": False},
        "vegetation filter only":                {"padding": 0, "veg_filter": True},
        "padding + vegetation filter":            {"padding": PADDING_PIXELS, "veg_filter": True},
    }

    results = {}

    for config_name, settings in configs.items():
        print(f"Testing: {config_name}")
        ious = []

        for item in precomputed:
            final_mask = refiner.refine_mask(
                item["t2_np"], item["candidate_np"], padding=settings["padding"]
            )

            if settings["veg_filter"]:
                final_mask = suppress_vegetation_false_positives(
                    final_mask, item["t1_np"], item["t2_np"]
                )

            metrics = compute_metrics(final_mask, item["gt_mask"])
            ious.append(metrics["iou"])

        avg_iou = np.mean(ious)
        results[config_name] = avg_iou
        print(f"  -> Average IoU: {avg_iou:.4f}\n")

    print("=" * 60)
    print("ABLATION RESULTS — Box Padding & Vegetation Filter")
    print("=" * 60)
    for name, iou in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name}: IoU = {iou:.4f}")


if __name__ == "__main__":
    main()
