"""
plot_lfe_single.py
Single graph with all three environments — matches Mingkang's Figure 1 style.
Usage: python plot_lfe_single.py
Output: ~/lfe_results/mingkang_baselines/lfe_single_graph.png
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_DIR  = os.path.expanduser("~/lfe_results/mingkang_baselines")
OUT_PATH = os.path.join(LOG_DIR, "lfe_single_graph.png")

ENVS = [
    ("FetchReach",        "reach.log",        "#2196F3"),  # blue
    ("FetchPush",         "push.log",         "#4CAF50"),  # green
    ("FetchPickAndPlace", "pickandplace.log",  "#F44336"),  # red
    ("FetchSlide",        "slide.log",         "#FF9800"),  # orange
]

def parse_log(path):
    epochs, rates = [], []
    epoch_pat = re.compile(r'\|\s*epoch\s*\|\s*(\d+)\s*\|')
    rate_pat  = re.compile(r'\|\s*test/success_rate\s*\|\s*([0-9.e+-]+)\s*\|')
    current_epoch = None
    with open(path, "r") as f:
        for line in f:
            em = epoch_pat.search(line)
            if em:
                current_epoch = int(em.group(1))
            rm = rate_pat.search(line)
            if rm and current_epoch is not None:
                epochs.append(current_epoch)
                rates.append(float(rm.group(1)))
                current_epoch = None
    return np.array(epochs), np.array(rates)

# ── Single figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for env_name, log_file, color in ENVS:
    log_path = os.path.join(LOG_DIR, log_file)
    if not os.path.exists(log_path):
        print(f"Skipping {env_name} — log not found")
        continue
    epochs, rates = parse_log(log_path)
    ax.plot(epochs, rates, color=color, linewidth=2.0, label=env_name)

ax.set_title("LFE Baseline — Learning Curves\n(50 epochs, 1 seed, 10 MPI workers)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=11)
ax.set_ylabel("Test Success Rate", fontsize=11)
ax.set_xlim(0, 49)
ax.set_ylim(0.0, 1.05)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
ax.legend(fontsize=10, loc="center right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Environment':<24} {'Final':>7} {'Best':>7}")
print("="*55)
for env_name, log_file, _ in ENVS:
    path = os.path.join(LOG_DIR, log_file)
    if not os.path.exists(path):
        continue
    epochs, rates = parse_log(path)
    print(f"{env_name:<24} {rates[-1]:>7.2f} {np.max(rates):>7.2f}")
print("="*55)
