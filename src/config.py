"""
Central configuration for geo_dinov2_sam2.
All tunable parameters live here — nowhere else in the codebase.
"""

import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DINOv2 settings
DINOV2_MODEL_NAME = "facebook/dinov2-base"
DINOV2_PATCH_SIZE = 14  # DINOv2-base uses 14x14 pixel patches internally

# SAM2 settings
SAM2_CHECKPOINT = "/home/ubuntu/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Change detection thresholds
CHANGE_THRESHOLD_PERCENTILE = 85  # top 15% most-different patches = candidate change

# Dataset settings
DATASET_NAME = "blanchon/LEVIR_CDPlus"
DATASET_SPLIT = "test"

# Paths
RESULTS_DIR = "/home/ubuntu/geo_dinov2_sam2/results"
MASKS_DIR = "/home/ubuntu/geo_dinov2_sam2/results/masks"
FIGURES_DIR = "/home/ubuntu/geo_dinov2_sam2/results/figures"
