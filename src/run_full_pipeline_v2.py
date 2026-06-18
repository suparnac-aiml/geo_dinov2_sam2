
"""
Final pipeline run: DINOv2-base + threshold 85 + SAM2 (no padding) +
vegetation filter, across the full 348-pair LEVIR-CD+ test set.
This produces the final reported metrics for the project.
"""

import sys
sys.path.append('.')
import json
import time
import numpy as np
from PIL import Image

from dataset import load_levircd, get_pair
from feature_extractor import DINOv2FeatureExtractor
from comparison import compute_change_heatmap
from thresholding import threshold_heatmap, upsample_to_image_size
from sam2_refiner import SAM2Refiner
from evaluation import prepare_ground_truth, compute_metrics
from vegetation_filter import suppress_vegetation_false_positives
from config import RESULTS_DIR, MASKS_DIR


def run_single_pair(index, dataset, extractor, refiner):
    t1, t2, gt_mask_pil = get_pair(dataset, index)

    features_t1, grid_size = extractor.extract_patch_features(t1)
    features_t2, _ = extractor.extract_patch_features(t2)

    heatmap = compute_change_heatmap(features_t1, features_t2, grid_size)
    binary_mask, threshold_val = threshold_heatmap(heatmap)
    upsampled = upsample_to_image_size(binary_mask, t1.size[::-1])

    t1_np = np.array(t1)
    t2_np = np.array(t2)

    final_mask = refiner.refine_mask(t2_np, upsampled.numpy(), padding=0)
    final_mask = suppress_vegetation_false_positives(final_mask, t1_np, t2_np)

    gt_mask = prepare_ground_truth(gt_mask_pil)
    metrics = compute_metrics(final_mask, gt_mask)
    metrics["pair_index"] = index

    return metrics, final_mask


def main():
    print("=" * 60)
    print("GeoSAM FINAL Pipeline v2 — with vegetation filter")
    print("=" * 60)

    dataset = load_levircd()
    total_pairs = len(dataset)

    extractor = DINOv2FeatureExtractor()
    refiner = SAM2Refiner()

    all_metrics = []
    failed_indices = []
    start_time = time.time()

    for i in range(total_pairs):
        try:
            metrics, final_mask = run_single_pair(i, dataset, extractor, refiner)
            all_metrics.append(metrics)

            mask_img = Image.fromarray((final_mask * 255).astype(np.uint8))
            mask_img.save(f"{MASKS_DIR}/pred_mask_v2_{i:04d}.png")

            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (total_pairs - i - 1)

            print(f"[{i+1}/{total_pairs}] IoU={metrics['iou']:.4f} "
                  f"F1={metrics['f1']:.4f} (~{remaining/60:.1f} min remaining)")

            if (i + 1) % 10 == 0:
                with open(f"{RESULTS_DIR}/metrics_v2_partial.json", "w") as f:
                    json.dump(all_metrics, f, indent=2)

        except Exception as e:
            print(f"[{i+1}/{total_pairs}] FAILED: {str(e)}")
            failed_indices.append(i)
            continue

    with open(f"{RESULTS_DIR}/metrics_v2_full.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    if len(all_metrics) > 0:
        avg_iou = np.mean([m["iou"] for m in all_metrics])
        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall = np.mean([m["recall"] for m in all_metrics])
        avg_f1 = np.mean([m["f1"] for m in all_metrics])

        summary = {
            "version": "v2_with_vegetation_filter",
            "total_pairs": total_pairs,
            "successful_pairs": len(all_metrics),
            "avg_iou": float(avg_iou),
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "avg_f1": float(avg_f1),
            "total_time_seconds": time.time() - start_time
        }

        with open(f"{RESULTS_DIR}/summary_v2.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\nFINAL v2 RESULTS")
        print(f"Average IoU:       {avg_iou:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall:    {avg_recall:.4f}")
        print(f"Average F1:        {avg_f1:.4f}")


if __name__ == "__main__":
    main()
