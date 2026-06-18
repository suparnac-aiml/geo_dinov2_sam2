import torch
import sys
sys.path.append('.')

from dataset import load_levircd, get_pair
from feature_extractor import DINOv2FeatureExtractor
from comparison import compute_change_heatmap
from thresholding import threshold_heatmap, upsample_to_image_size

ds = load_levircd()
t1, t2, mask = get_pair(ds, 0)

extractor = DINOv2FeatureExtractor()
features_t1, grid_size = extractor.extract_patch_features(t1)
features_t2, _ = extractor.extract_patch_features(t2)

heatmap = compute_change_heatmap(features_t1, features_t2, grid_size)
binary_mask, threshold_val = threshold_heatmap(heatmap)

print(f"Threshold value used: {threshold_val:.4f}")
print(f"Candidate change patches: {binary_mask.sum().item():.0f} / {binary_mask.numel()}")

upsampled = upsample_to_image_size(binary_mask, (1024, 1024))
print(f"Upsampled shape: {upsampled.shape}")
print(f"Upsampled unique values: {torch.unique(upsampled)}")


import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(t1)
axes[0].set_title("T1 (before)")
axes[0].axis('off')

axes[1].imshow(t2)
axes[1].set_title("T2 (after)")
axes[1].axis('off')

axes[2].imshow(heatmap.numpy(), cmap='hot')
axes[2].set_title("DINOv2 Change Heatmap (16x16)")
axes[2].axis('off')

axes[3].imshow(upsampled.numpy(), cmap='gray')
axes[3].set_title(f"Thresholded Mask (top 15%)")
axes[3].axis('off')

plt.tight_layout()
plt.savefig('/home/ubuntu/geo_dinov2_sam2/results/figures/test_pair0_visual.png', dpi=100)
print("Saved visual to results/figures/test_pair0_visual.png")
