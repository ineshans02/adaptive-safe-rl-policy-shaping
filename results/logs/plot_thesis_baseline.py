"""
plot_thesis_baseline.py
Plots multi-seed learning curves for thesis baseline (LFE, FetchPush).
Matches Mingkang's Figure 1 style: mean ± std + stderr shading.
Usage: python plot_thesis_baseline.py
Output: ~/lfe_results/thesis_baseline/push/thesis_baseline_push.png
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
LOG_DIR  = os.path.expanduser("~/lfe_results/thesis_baseline/push")
OUT_PATH = os.path.join(LOG_DIR, "thesis_baseline_push.png")
SEEDS    = [0, 1, 2, 3, 4]

LFE_COLOR = "#2ca02c"   # green — matches Mingkang's Figure 1

# ── Parser ────────────────────────────────────────────────────────────────────
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

# ── Load all available seeds ──────────────────────────────────────────────────
all_rates = []
available_seeds = []

for seed in SEEDS:
    log_path = os.path.join(LOG_DIR, f"seed_{seed}.log")
    if not os.path.exists(log_path):
        print(f"Seed {seed}: log not found — skipping")
        continue
    epochs, rates = parse_log(log_path)
    if len(rates) == 0:
        print(f"Seed {seed}: no data — skipping")
        continue
    all_rates.append(rates)
    available_seeds.append(seed)
    print(f"Seed {seed}: {len(rates)} epochs, "
          f"final={rates[-1]:.2f}, best={np.max(rates):.2f}")

print(f"\nAvailable seeds: {available_seeds}")

# ── Align to shortest run ─────────────────────────────────────────────────────
min_len = min(len(r) for r in all_rates)
all_rates = np.array([r[:min_len] for r in all_rates])
epochs    = np.arange(min_len)

# ── Compute statistics ────────────────────────────────────────────────────────
mean   = np.mean(all_rates, axis=0)
std    = np.std(all_rates, axis=0)
stderr = std / np.sqrt(len(available_seeds))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

# Light shading — std (like Mingkang's Figure 1)
ax.fill_between(epochs,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                alpha=0.15, color=LFE_COLOR, label="_nolegend_")

# Dark shading — stderr
ax.fill_between(epochs,
                np.clip(mean - stderr, 0, 1),
                np.clip(mean + stderr, 0, 1),
                alpha=0.30, color=LFE_COLOR, label="_nolegend_")

# Mean line
ax.plot(epochs, mean,
        color=LFE_COLOR, linewidth=2.0,
        label=f"LFE (mean ± std, n={len(available_seeds)} seeds)")

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_title("LFE Baseline — FetchPush-v1\n"
             f"({min_len} epochs, {len(available_seeds)} seeds, 19 MPI workers)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=11)
ax.set_ylabel("Test Success Rate", fontsize=11)
ax.set_xlim(0, min_len - 1)
ax.set_ylim(0.0, 1.05)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
ax.legend(fontsize=10, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Summary annotation ────────────────────────────────────────────────────────
final_mean = mean[-1]
final_std  = std[-1]
best_mean  = np.max(mean)
ax.text(0.97, 0.05,
        f"Final: {final_mean:.2f} ± {final_std:.2f}  |  Best mean: {best_mean:.2f}",
        transform=ax.transAxes,
        fontsize=9, ha="right", color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"FetchPush LFE Baseline — Multi-seed Summary")
print("="*55)
print(f"Seeds available : {available_seeds}")
print(f"Epochs per seed : {min_len}")
print(f"Final mean      : {final_mean:.3f} ± {final_std:.3f}")
print(f"Best mean epoch : {np.argmax(mean)}")
print(f"Best mean rate  : {best_mean:.3f}")
print("="*55)
