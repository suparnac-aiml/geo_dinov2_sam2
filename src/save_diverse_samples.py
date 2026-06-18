"""
Pulls a diverse set of LEVIR-CD+ pairs for demo purposes — chosen
based on their actual IoU scores from our v2 evaluation run, so we
can show a realistic spread: strong success, moderate, and honest
failure cases.
"""

import json
from datasets import load_dataset

with open("/home/ubuntu/geo_dinov2_sam2/results/metrics_v2_full.json", "r") as f:
    all_metrics = json.load(f)

# Sort by IoU to find genuine high/low/mid examples
sorted_metrics = sorted(all_metrics, key=lambda m: m["iou"])

# Pick a spread: worst, low-mid, high-mid, best
worst = sorted_metrics[0]
low_mid = sorted_metrics[len(sorted_metrics) // 4]
high_mid = sorted_metrics[3 * len(sorted_metrics) // 4]
best = sorted_metrics[-1]

selected = {
    "worst_case": worst,
    "low_case": low_mid,
    "high_case": high_mid,
    "best_case": best
}

print("Selected demo pairs:")
for label, m in selected.items():
    print(f"  {label}: index={m['pair_index']}, IoU={m['iou']:.4f}")

ds = load_dataset("blanchon/LEVIR_CDPlus", split="test")

for label, m in selected.items():
    idx = m["pair_index"]
    ds[idx]["image1"].save(f"/tmp/demo_{label}_t1.png")
    ds[idx]["image2"].save(f"/tmp/demo_{label}_t2.png")
    print(f"Saved /tmp/demo_{label}_t1.png and t2.png (IoU={m['iou']:.4f})")

print("\nDone. All demo pairs saved to /tmp/")
