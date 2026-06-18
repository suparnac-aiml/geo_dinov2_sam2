"""
Full batch pipeline: runs DINOv2 + SAM2 change detection across
the entire LEVIR-CD+ test set (348 pairs), saves per-pair metrics
and masks, and computes dataset-wide aggregate statistics.

Designed to survive partial failures — if one pair errors, we log
it and continue, rather than crashing the entire multi-hour run.
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
from config import RESULTS_DIR, MASKS_DIR


def run_single_pair(index, dataset, extractor, refiner):
    """
    Runs the full pipeline on one pair and returns its metrics.
    Raises an exception on failure — caller handles logging/skipping.
    """
    t1, t2, gt_mask_pil = get_pair(dataset, index)

    features_t1, grid_size = extractor.extract_patch_features(t1)
    features_t2, _ = extractor.extract_patch_features(t2)

    heatmap = compute_change_heatmap(features_t1, features_t2, grid_size)
    binary_mask, threshold_val = threshold_heatmap(heatmap)
    upsampled = upsample_to_image_size(binary_mask, t1.size[::-1])  # (H, W)

    t2_np = np.array(t2)
    final_mask = refiner.refine_mask(t2_np, upsampled.numpy())

    gt_mask = prepare_ground_truth(gt_mask_pil)
    metrics = compute_metrics(final_mask, gt_mask)
    metrics["threshold_used"] = float(threshold_val)
    metrics["pair_index"] = index

    return metrics, final_mask


def main():
    print("=" * 60)
    print("GeoSAM Full Pipeline — LEVIR-CD+ Test Set Evaluation")
    print("=" * 60)

    dataset = load_levircd()
    total_pairs = len(dataset)
    print(f"Total pairs to process: {total_pairs}")

    print("\nLoading models (one-time cost)...")
    extractor = DINOv2FeatureExtractor()
    refiner = SAM2Refiner()
    print("Models loaded. Starting batch processing.\n")

    all_metrics = []
    failed_indices = []
    start_time = time.time()

    for i in range(total_pairs):
        pair_start = time.time()
        try:
            metrics, final_mask = run_single_pair(i, dataset, extractor, refiner)
            all_metrics.append(metrics)

            # Save the predicted mask as an image for later inspection
            mask_img = Image.fromarray((final_mask * 255).astype(np.uint8))
            mask_img.save(f"{MASKS_DIR}/pred_mask_{i:04d}.png")

            pair_time = time.time() - pair_start
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (total_pairs - i - 1)

            print(
                f"[{i+1}/{total_pairs}] IoU={metrics['iou']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"({pair_time:.1f}s/pair, "
                f"~{remaining/60:.1f} min remaining)"
            )

            # Save progress incrementally — if the script crashes later,
            # we don't lose everything computed so far
            if (i + 1) % 10 == 0:
                with open(f"{RESULTS_DIR}/metrics_partial.json", "w") as f:
                    json.dump(all_metrics, f, indent=2)

        except Exception as e:
            print(f"[{i+1}/{total_pairs}] FAILED: {str(e)}")
            failed_indices.append(i)
            continue

    # Final save
    with open(f"{RESULTS_DIR}/metrics_full.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(f"{RESULTS_DIR}/failed_indices.json", "w") as f:
        json.dump(failed_indices, f, indent=2)

    # Aggregate statistics
    if len(all_metrics) > 0:
        avg_iou = np.mean([m["iou"] for m in all_metrics])
        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall = np.mean([m["recall"] for m in all_metrics])
        avg_f1 = np.mean([m["f1"] for m in all_metrics])

        summary = {
            "total_pairs": total_pairs,
            "successful_pairs": len(all_metrics),
            "failed_pairs": len(failed_indices),
            "avg_iou": float(avg_iou),
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "avg_f1": float(avg_f1),
            "total_time_seconds": time.time() - start_time
        }

        with open(f"{RESULTS_DIR}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Successful: {len(all_metrics)}/{total_pairs}")
        print(f"Average IoU:       {avg_iou:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall:    {avg_recall:.4f}")
        print(f"Average F1:        {avg_f1:.4f}")
        print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")
    else:
        print("No pairs processed successfully — check failed_indices.json")


if __name__ == "__main__":
    main()
