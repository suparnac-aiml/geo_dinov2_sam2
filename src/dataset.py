"""
Dataset loading and access for LEVIR-CD+.
Wraps the HuggingFace dataset object into something our pipeline
can index cleanly by integer position.
"""

from datasets import load_dataset
from config import DATASET_NAME, DATASET_SPLIT


def load_levircd():
    """
    Loads the LEVIR-CD+ test split from HuggingFace.

    Returns:
        A HuggingFace Dataset object with fields:
        'image1' (T1 PIL Image), 'image2' (T2 PIL Image),
        'mask' (ground truth PIL Image, grayscale)
    """
    print(f"Loading {DATASET_NAME} ({DATASET_SPLIT} split)...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"Loaded {len(ds)} image pairs.")
    return ds


def get_pair(dataset, index):
    """
    Retrieves a single T1/T2/mask triplet by index.

    Args:
        dataset: the loaded HuggingFace dataset
        index: integer position (0 to len(dataset)-1)

    Returns:
        tuple of (image_t1, image_t2, ground_truth_mask) as PIL Images
    """
    sample = dataset[index]
    return sample["image1"], sample["image2"], sample["mask"]
