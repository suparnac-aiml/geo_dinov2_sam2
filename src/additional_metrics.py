"""
Computes additional analysis metrics from the saved per-pair results:
- mIoU specifically over the "changed" class (clarifying what our
  reported IoU actually represents)
- Percentage of image pairs achieving IoU > various thresholds
"""

import json
import numpy as np

with open("/home/ubuntu/geo_dinov2_sam2/results/metrics_v2_full.json", "r") as f:
    all_metrics = json.load(f)

ious = np.array([m["iou"] for m in all_metrics])
precisions = np.array([m["precision"] for m in all_metrics])
recalls = np.array([m["recall"] for m in all_metrics])
f1s = np.array([m["f1"] for m in all_metrics])

print(f"Total pairs evaluated: {len(all_metrics)}")
print()
print(f"Mean IoU (changed class):  {ious.mean():.4f}")
print(f"Mean Precision:            {precisions.mean():.4f}")
print(f"Mean Recall:               {recalls.mean():.4f}")
print(f"Mean F1/Dice:              {f1s.mean():.4f}")
print()
print(f"Median IoU:                {np.median(ious):.4f}")
print(f"Std deviation of IoU:      {ious.std():.4f}")
print(f"Min IoU:                   {ious.min():.4f}")
print(f"Max IoU:                   {ious.max():.4f}")
print()

thresholds = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
for t in thresholds:
    pct = (ious > t).sum() / len(ious) * 100
    print(f"Pairs with IoU > {t}: {(ious > t).sum()}/{len(ious)} ({pct:.1f}%)")
