"""
Quick standalone test: confirm DINOv2 feature extraction
produces sensible output shapes before building the rest
of the pipeline on top of it.
"""

import sys
sys.path.append('.')

from dataset import load_levircd, get_pair
from feature_extractor import DINOv2FeatureExtractor

ds = load_levircd()
t1, t2, mask = get_pair(ds, 0)

extractor = DINOv2FeatureExtractor()

features_t1, grid_size = extractor.extract_patch_features(t1)
features_t2, _ = extractor.extract_patch_features(t2)

print(f"T1 feature shape: {features_t1.shape}")
print(f"T2 feature shape: {features_t2.shape}")
print(f"Grid size: {grid_size}")
print(f"T1 image size: {t1.size}")
print(f"Mask size: {mask.size}")
