"""
GeoSAM Streamlit Demo App.
Upload two satellite images (before/after) and visualize zero-shot
change detection using DINOv2 + SAM2.
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from feature_extractor import DINOv2FeatureExtractor
from comparison import compute_change_heatmap
from thresholding import threshold_heatmap, upsample_to_image_size
from sam2_refiner import SAM2Refiner
from vegetation_filter import suppress_vegetation_false_positives

st.set_page_config(page_title="GeoSAM — Zero-Shot Change Detection", layout="wide")

st.title("🛰️ GeoSAM: Zero-Shot Satellite Change Detection")
st.markdown("""
Upload two satellite images of the **same location** at different times.
This tool uses **DINOv2** (feature extraction) + **SAM2** (segmentation) —
both frozen, zero-shot, no training on satellite data — to detect change.
""")

# Cache model loading so it only happens once per session, not per upload
@st.cache_resource
def load_models():
    extractor = DINOv2FeatureExtractor()
    refiner = SAM2Refiner()
    return extractor, refiner

extractor, refiner = load_models()

col1, col2 = st.columns(2)
with col1:
    file_t1 = st.file_uploader("Upload T1 (before)", type=["png", "jpg", "jpeg"])
with col2:
    file_t2 = st.file_uploader("Upload T2 (after)", type=["png", "jpg", "jpeg"])

use_vegetation_filter = st.checkbox("Apply vegetation false-positive filter", value=True)

if file_t1 and file_t2:
    t1 = Image.open(file_t1).convert("RGB")
    t2 = Image.open(file_t2).convert("RGB")

    if t1.size != t2.size:
        st.error(f"Image size mismatch: T1 is {t1.size}, T2 is {t2.size}. "
                 f"Both images must be the same size (co-registered).")
    else:
        with st.spinner("Running DINOv2 feature extraction..."):
            features_t1, grid_size = extractor.extract_patch_features(t1)
            features_t2, _ = extractor.extract_patch_features(t2)
            heatmap = compute_change_heatmap(features_t1, features_t2, grid_size)
            binary_mask, threshold_val = threshold_heatmap(heatmap)
            upsampled = upsample_to_image_size(binary_mask, t1.size[::-1])

        with st.spinner("Running SAM2 segmentation refinement..."):
            t1_np = np.array(t1)
            t2_np = np.array(t2)
            final_mask = refiner.refine_mask(t2_np, upsampled.numpy(), padding=0)

            if use_vegetation_filter:
                final_mask = suppress_vegetation_false_positives(final_mask, t1_np, t2_np)

        changed_pixels = int(final_mask.sum())
        total_pixels = final_mask.size
        pct_changed = (changed_pixels / total_pixels) * 100

        st.success("Pipeline complete!")

        m1, m2, m3 = st.columns(3)
        m1.metric("Changed Pixels", f"{changed_pixels:,}")
        m1.metric("Total Pixels", f"{total_pixels:,}")
        m2.metric("Percent Changed", f"{pct_changed:.2f}%")
        m3.metric("DINOv2 Threshold Used", f"{threshold_val:.3f}")

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(t1)
        axes[0].set_title("T1 (before)")
        axes[0].axis('off')

        axes[1].imshow(t2)
        axes[1].set_title("T2 (after)")
        axes[1].axis('off')

        axes[2].imshow(heatmap.numpy(), cmap='hot')
        axes[2].set_title("DINOv2 Change Heatmap")
        axes[2].axis('off')

        overlay = t2_np.copy()
        overlay[final_mask == 1] = [255, 0, 0]
        axes[3].imshow(overlay)
        axes[3].set_title("Change Overlay (red = detected change)")
        axes[3].axis('off')

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        ---
        **About this demo:** This pipeline is zero-shot — DINOv2 and SAM2
        were never trained on satellite imagery or this specific task.
        Performance is strongest on large, high-contrast changes
        (e.g. new construction, deforestation) and weaker on small,
        subtle changes. See the
        [GitHub repository](https://github.com/YOUR_USERNAME/geo_dinov2_sam2)
        for full evaluation metrics and ablation studies.
        """)
else:
    st.info("Upload both T1 and T2 images to begin.")
