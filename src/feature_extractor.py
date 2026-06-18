"""
DINOv2 feature extraction.
Converts an image into a grid of patch embeddings describing
visual appearance at each spatial location.
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from config import DINOV2_MODEL_NAME, DEVICE


class DINOv2FeatureExtractor:
    def __init__(self):
        print(f"Loading DINOv2 model: {DINOV2_MODEL_NAME}")
        self.processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME)
        self.model = AutoModel.from_pretrained(DINOV2_MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()  # inference mode — disables dropout etc.

    @torch.no_grad()  # we never need gradients — saves memory and time
    def extract_patch_features(self, image):
        """
        Args:
            image: a PIL Image (RGB)

        Returns:
            patch_features: tensor of shape (num_patches, 768)
            grid_size: (height_patches, width_patches) tuple,
                       needed later to reshape back into a 2D map
        """
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)

        # outputs.last_hidden_state shape: (1, num_tokens, 768)
        # token 0 is the [CLS] token (whole-image summary) — we discard it
        # remaining tokens are the actual spatial patches
        patch_tokens = outputs.last_hidden_state[0, 1:, :]  # (num_patches, 768)

        # DINOv2-base processes 518x518 images by default into 37x37 patches
        num_patches = patch_tokens.shape[0]
        grid_side = int(num_patches ** 0.5)  # assumes square grid

        return patch_tokens, (grid_side, grid_side)
