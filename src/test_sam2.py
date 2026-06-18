import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_levircd, get_pair
from feature_extractor import DINOv2FeatureExtractor
from comparison import compute_change_heatmap
from thresholding import threshold_heatmap, upsample_to_image_size
from sam2_refiner import SAM2Refiner

ds = load_levircd()
t1, t2, mask = get_pair(ds, 0)

extractor = DINOv2FeatureExtractor()
features_t1, grid_size = extractor.extract_patch_features(t1)
features_t2, _ = extractor.extract_patch_features(t2)

heatmap = compute_change_heatmap(features_t1, features_t2, grid_size)
binary_mask, threshold_val = threshold_heatmap(heatmap)
upsampled = upsample_to_image_size(binary_mask, (1024, 1024))

print("Running SAM2 refinement...")
refiner = SAM2Refiner()

t2_np = np.array(t2)
candidate_np = upsampled.numpy()

final_mask = refiner.refine_mask(t2_np, candidate_np)

print(f"Final mask shape: {final_mask.shape}")
print(f"Final mask changed pixels: {final_mask.sum()}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(t2)
axes[0].set_title("T2 (after)")
axes[0].axis('off')

axes[1].imshow(candidate_np, cmap='gray')
axes[1].set_title("DINOv2 candidate regions (blocky)")
axes[1].axis('off')

axes[2].imshow(final_mask, cmap='gray')
axes[2].set_title("SAM2-refined mask (precise)")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('/home/ubuntu/geo_dinov2_sam2/results/figures/test_pair0_sam2.png', dpi=100)
print("Saved to results/figures/test_pair0_sam2.png")
