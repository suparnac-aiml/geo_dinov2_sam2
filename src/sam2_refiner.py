"""
Converts thresholded DINOv2 candidate regions into SAM2 prompts,
runs SAM2 to get precise segmentation masks, and merges results
into one final change mask.
"""

import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from config import SAM2_CHECKPOINT, SAM2_CONFIG, DEVICE


class SAM2Refiner:
    def __init__(self):
        print("Loading SAM2 model...")
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2 loaded.")

    def get_bounding_boxes(self, binary_mask_np, padding=0):
        """
        Finds connected white regions in the binary mask and
        converts each into a bounding box (x_min, y_min, x_max, y_max).

        Args:
            binary_mask_np: 2D numpy array (H, W), values 0 or 1
            padding: pixels to expand each box by on all sides

        Returns:
            list of bounding boxes, one per connected region
        """
        mask_uint8 = (binary_mask_np * 255).astype(np.uint8)
        h, w = binary_mask_np.shape

        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw > 10 and bh > 10:
                x_min = max(0, x - padding)
                y_min = max(0, y - padding)
                x_max = min(w, x + bw + padding)
                y_max = min(h, y + bh + padding)
                boxes.append([x_min, y_min, x_max, y_max])

        return boxes

    def refine_mask(self, image_t2_np, candidate_mask_np, padding=0):
        """
        Args:
            image_t2_np: numpy array (H, W, 3) — the T2 image
            candidate_mask_np: numpy array (H, W) — upsampled binary mask
            padding: pixels to expand each candidate box by

        Returns:
            final_mask: numpy array (H, W), values 0 or 1
        """
        boxes = self.get_bounding_boxes(candidate_mask_np, padding=padding)

        if len(boxes) == 0:
            return np.zeros_like(candidate_mask_np)

        self.predictor.set_image(image_t2_np)
        final_mask = np.zeros_like(candidate_mask_np)

        for box in boxes:
            box_array = np.array(box)
            masks, scores, _ = self.predictor.predict(
                box=box_array,
                multimask_output=False
            )
            final_mask = np.logical_or(final_mask, masks[0]).astype(np.uint8)

        return final_mask
