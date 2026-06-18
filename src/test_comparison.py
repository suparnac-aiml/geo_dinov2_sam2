import sys
sys.path.append('.')

from dataset import load_levircd, get_pair
from feature_extractor import DINOv2FeatureExtractor
from comparison import compute_change_heatmap

ds = load_levircd()
t1, t2, mask = get_pair(ds, 0)

extractor = DINOv2FeatureExtractor()
features_t1, grid_size = extractor.extract_patch_features(t1)
features_t2, _ = extractor.extract_patch_features(t2)

heatmap = compute_change_heatmap(features_t1, features_t2, grid_size)

print(f"Heatmap shape: {heatmap.shape}")
print(f"Heatmap min: {heatmap.min().item():.4f}")
print(f"Heatmap max: {heatmap.max().item():.4f}")
print(f"Heatmap mean: {heatmap.mean().item():.4f}")
