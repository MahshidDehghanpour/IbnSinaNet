"""
statistical_analysis.py
========================
اجرا:
    python statistical_analysis.py

ورودی:  all_results.json
خروجی: p-value هر کانفیگ در مقایسه با Full IbnSinaNet
"""

import json
import numpy as np
from scipy.stats import wilcoxon

# ─── لود نتایج ───────────────────────────────────────────────────────────
with open("all_results.json", "r") as f:
    data = json.load(f)

ablation = data["ablation"]

# ─── Dice scores مدل کامل ────────────────────────────────────────────────
full_dice = np.array([r["dice"] for r in ablation["full"]["results"]])

# ─── مقایسه هر کانفیگ با Full IbnSinaNet ─────────────────────────────────
configs = [
    {"tag": "baseline",     "name": "Baseline U-Net"},
    {"tag": "dense",        "name": "+DenseNet"},
    {"tag": "dense_csp",    "name": "+DenseNet +CSP"},
    {"tag": "dense_csp_sa", "name": "+DenseNet +CSP +SA"},
]

print("\nWilcoxon Signed-Rank Test (vs Full IbnSinaNet)")
print("=" * 60)
print(f"  {'Configuration':<30} {'Δ Dice':>8}  {'p-value':>10}  {'Sig':>5}")
print("  " + "-"*55)

for config in configs:
    tag  = config["tag"]
    name = config["name"]

    other_dice = np.array([r["dice"] for r in ablation[tag]["results"]])

    n       = min(len(full_dice), len(other_dice))
    fd      = full_dice[:n]
    od      = other_dice[:n]

    stat, p = wilcoxon(fd, od)
    delta   = (fd.mean() - od.mean()) * 100
    sig     = "✓" if p < 0.05 else "✗"

    print(f"  {name:<30} {delta:>+7.2f}%  {p:>10.4f}  {sig:>5}")

print("\n  p < 0.05 = معنادار آماری")
