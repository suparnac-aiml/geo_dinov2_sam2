# GeoSAM: Zero-Shot Satellite Change Detection with DINOv2 + SAM2

Zero-shot geospatial change detection pipeline combining DINOv2 (frozen feature extraction) and SAM2 (segmentation refinement) — no training on satellite imagery required.

## Overview

GeoSAM detects land-cover change between two satellite images of the same location taken at different times, using foundation models never trained on satellite data. The pipeline compares DINOv2 patch embeddings between image pairs to localize candidate change regions, then uses SAM2 to refine these into pixel-accurate segmentation masks.

## Results

Evaluated on the full LEVIR-CD+ test set (348 image pairs):

| Metric | Baseline (v1) | + Vegetation Filter (v2) |
|---|---|---|
| IoU | 0.0791 | 0.0822 |
| Precision | 0.0985 | 0.1029 |
| Recall | 0.3996 | 0.3981 |
| F1/Dice | 0.1307 | 0.1359 |

**Performance is strongly scale-dependent**: Pearson correlation between ground-truth change size and IoU is **0.58**. Splitting at median change size, large-change pairs achieve **0.151 mean IoU** vs **0.014** for small-change pairs — a 10.8x difference.

This aligns with published literature noting plain zero-shot DINOv2+SAM baselines underperform specialized zero-shot methods (e.g., AnyChange, arXiv:2402.01188), while supervised SOTA models (ChangeFormer, PeftCD) achieve 80-86% IoU on this same benchmark with full training.

## Ablation Studies

**1. Threshold sensitivity** (percentile-based thresholding, tested 70-95th percentile on 30-pair subset): 85th percentile was near-optimal (IoU=0.096), confirming our default setting.

**2. Model scale** (DINOv2-base vs DINOv2-large): Base outperformed large (0.096 vs 0.075 IoU) — larger models trained on natural images don't necessarily transfer better zero-shot to an unfamiliar domain.

**3. SAM2 box padding + vegetation filtering**: Padding consistently hurt performance (likely causing SAM2 to over-segment into unchanged surrounding context). A simple RGB-based vegetation heuristic (suppressing persistent-green regions) gave a small but consistent improvement (+4% IoU).

## Architecture
T1, T2 images → DINOv2 (frozen) → patch embeddings (256×768 each)

→ cosine similarity → change heatmap (16×16)

→ percentile threshold → candidate regions

→ SAM2 (boxes prompts) → precise segmentation

→ vegetation filter → final change mask

→ evaluation vs ground truth (IoU, F1, Precision, Recall)

## Tech Stack

- **DINOv2** (facebook/dinov2-base) — frozen feature extraction
- **SAM2** (sam2.1_hiera_base_plus) — segmentation refinement
- **Dataset**: LEVIR-CD+ (348 test pairs, via HuggingFace `blanchon/LEVIR_CDPlus`)
- **Infrastructure**: AWS EC2 (CPU-only, m7i-flex.large), Ubuntu 24.04
- **Deployment**: Streamlit

## Setup

```bash
python3 -m venv geosam-env
source geosam-env/bin/activate
pip install -r requirements.txt
```

## Usage

Run full evaluation:
```bash
python3 src/run_full_pipeline_v2.py
```

Launch demo app:
```bash
streamlit run src/app.py
```

## Limitations

- Zero-shot performance is well below supervised SOTA (expected — see Results)
- Vegetation filter uses RGB greenness heuristic, not true NDVI (no NIR band available in this RGB-only dataset)
- Performance degrades significantly on small/subtle changes due to DINOv2's fixed 64×64px patch granularity at this resolution

## Future Work

- Multi-scale patch comparison to address the size-correlation limitation
- True NDVI-based filtering with multispectral data (e.g., Sentinel-2)
- Lightweight fine-tuning (LoRA/PEFT) on a small labeled subset to bridge the zero-shot to supervised gap

## Author

Suparna C — [LinkedIn](www.linkedin.com/in/suparna-c-279339ab) | [GitHub](https://github.com/suparnac-aiml)
