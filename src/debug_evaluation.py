import sys
sys.path.append('.')
import numpy as np

from dataset import load_levircd, get_pair
from feature_extractor import DINOv2FeatureExtractor
from comparison import compute_change_heatmap
from thresholding import threshold_heatmap, upsample_to_image_size
from sam2_refiner import SAM2Refiner
from evaluation import prepare_ground_truth

ds = load_levircd()
t1, t2, mask = get_pair(ds, 0)

extractor = DINOv2FeatureExtractor()
features_t1, grid_size = extractor.extract_patch_features(t1)
features_t2, _ = extractor.extract_patch_features(t2)

heatmap = compute_change_heatmap(features_t1, features_t2, grid_size)
binary_mask, _ = threshold_heatmap(heatmap)
upsampled = upsample_to_image_size(binary_mask, (1024, 1024))

refiner = SAM2Refiner()
t2_np = np.array(t2)
final_mask = refiner.refine_mask(t2_np, upsampled.numpy())

gt_mask = prepare_ground_truth(mask)

print(f"final_mask shape: {final_mask.shape}, dtype: {final_mask.dtype}")
print(f"final_mask unique values: {np.unique(final_mask)}")
print(f"final_mask sum: {final_mask.sum()}")
print()
print(f"gt_mask shape: {gt_mask.shape}, dtype: {gt_mask.dtype}")
print(f"gt_mask unique values: {np.unique(gt_mask)}")
print(f"gt_mask sum: {gt_mask.sum()}")
print()
print(f"Raw mask PIL mode: {mask.mode}, size: {mask.size}")
print(f"Raw mask numpy unique values (before threshold): {np.unique(np.array(mask))}")
