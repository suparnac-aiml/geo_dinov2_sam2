"""
Tests DINOv2-large (instead of base) on the same fixed 30-pair subset,
using the best threshold (85) found in the previous experiment, to see
if a larger feature extractor improves IoU.
"""

import sys
sys.path.append('.')
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from dataset import load_levircd, get_pair
from thresholding import upsample_to_image_size
from sam2_refiner import SAM2Refiner
from evaluation import prepare_ground_truth, compute_metrics
from config import DEVICE

SUBSET_SIZE = 30
BEST_THRESHOLD_PCT = 85
LARGE_MODEL_NAME = "facebook/dinov2-large"


class DINOv2LargeExtractor:
    """Same logic as our existing DINOv2FeatureExtractor class,
    but pointed at the 'large' checkpoint instead of 'base'."""

    def __init__(self):
        print(f"Loading {LARGE_MODEL_NAME} (this is bigger, will take longer)...")
        self.processor = AutoImageProcessor.from_pretrained(LARGE_MODEL_NAME)
        self.model = AutoModel.from_pretrained(LARGE_MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def extract_patch_features(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        patch_tokens = outputs.last_hidden_state[0, 1:, :]
        num_patches = patch_tokens.shape[0]
        grid_side = int(num_patches ** 0.5)
        return patch_tokens, (grid_side, grid_side)


def compute_change_heatmap(features_t1, features_t2, grid_size):
    import torch.nn.functional as F
    similarity = F.cosine_similarity(features_t1, features_t2, dim=1)
    distance = 1 - similarity
    grid_h, grid_w = grid_size
    return distance.reshape(grid_h, grid_w)


def threshold_at_percentile(heatmap, percentile):
    flat_values = heatmap.flatten().numpy()
    threshold_value = np.percentile(flat_values, percentile)
    return (heatmap > threshold_value).float()


def main():
    dataset = load_levircd()
    subset_indices = list(range(SUBSET_SIZE))  # same fixed 30 pairs as before

    extractor = DINOv2LargeExtractor()
    refiner = SAM2Refiner()

    ious = []
    print(f"\nRunning DINOv2-large on {SUBSET_SIZE} pairs, threshold={BEST_THRESHOLD_PCT}...\n")

    for idx in subset_indices:
        t1, t2, gt_mask_pil = get_pair(dataset, idx)

        feat_t1, grid_size = extractor.extract_patch_features(t1)
        feat_t2, _ = extractor.extract_patch_features(t2)

        heatmap = compute_change_heatmap(feat_t1, feat_t2, grid_size)
        binary_mask = threshold_at_percentile(heatmap, BEST_THRESHOLD_PCT)
        upsampled = upsample_to_image_size(binary_mask, t1.size[::-1])

        t2_np = np.array(t2)
        final_mask = refiner.refine_mask(t2_np, upsampled.numpy())

        gt_mask = prepare_ground_truth(gt_mask_pil)
        metrics = compute_metrics(final_mask, gt_mask)
        ious.append(metrics["iou"])

        print(f"[{idx+1}/{SUBSET_SIZE}] IoU={metrics['iou']:.4f}")

    avg_iou = np.mean(ious)
    print(f"\n{'='*50}")
    print(f"DINOv2-LARGE avg IoU on subset: {avg_iou:.4f}")
    print(f"DINOv2-BASE avg IoU on same subset (previous): 0.0957")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
