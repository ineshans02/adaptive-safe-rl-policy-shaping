"""
plot_lfe_baseline.py
Parses training logs and generates learning curve plots for
Mingkang's LFE baseline across FetchReach, FetchPush, FetchPickAndPlace.
Usage: python plot_lfe_baseline.py
Output: ~/lfe_results/mingkang_baselines/lfe_baseline_curves.png
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ─────────────────────────────────────────────────────────────
LOG_DIR  = os.path.expanduser("~/lfe_results/mingkang_baselines")
OUT_PATH = os.path.join(LOG_DIR, "lfe_baseline_curves.png")

ENVS = [
    ("FetchReach",        "reach.log",        "#2196F3"),   # blue
    ("FetchPush",         "push.log",         "#4CAF50"),   # green
    ("FetchPickAndPlace", "pickandplace.log",  "#F44336"),   # red
]

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_log(path):
    """Extract (epoch, test_success_rate) pairs from a training log."""
    epochs, rates = [], []
    epoch_pattern = re.compile(r'\|\s*epoch\s*\|\s*(\d+)\s*\|')
    rate_pattern  = re.compile(r'\|\s*test/success_rate\s*\|\s*([0-9.e+-]+)\s*\|')

    current_epoch = None
    with open(path, "r") as f:
        for line in f:
            em = epoch_pattern.search(line)
            if em:
                current_epoch = int(em.group(1))
            rm = rate_pattern.search(line)
            if rm and current_epoch is not None:
                epochs.append(current_epoch)
                rates.append(float(rm.group(1)))
                current_epoch = None   # reset until next epoch line

    return np.array(epochs), np.array(rates)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle("Mingkang LFE Baseline — Learning Curves\n(1 seed, 10 MPI workers)",
             fontsize=14, fontweight="bold", y=1.02)

for ax, (env_name, log_file, color) in zip(axes, ENVS):
    log_path = os.path.join(LOG_DIR, log_file)
    epochs, rates = parse_log(log_path)

    # ── Smoothed line (rolling window of 5 epochs) ──
    if len(rates) >= 5:
        kernel  = np.ones(5) / 5
        smooth  = np.convolve(rates, kernel, mode='valid')
        smooth_epochs = epochs[2: 2 + len(smooth)]
    else:
        smooth, smooth_epochs = rates, epochs

    # ── Raw data (faint) ──
    ax.plot(epochs, rates, color=color, alpha=0.25, linewidth=1.0, label="_nolegend_")

    # ── Smoothed data (bold) ──
    ax.plot(smooth_epochs, smooth, color=color, linewidth=2.5, label="LFE (smoothed)")

    # ── Best success rate marker ──
    best_idx  = np.argmax(rates)
    best_rate = rates[best_idx]
    best_ep   = epochs[best_idx]
    ax.scatter([best_ep], [best_rate], color=color, s=80, zorder=5,
               label=f"Best: {best_rate:.2f} @ ep {best_ep}")

    # ── Formatting ──
    ax.set_title(env_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Test Success Rate", fontsize=11)
    ax.set_xlim(0, max(epochs) + 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Annotate final value ──
    final_rate = rates[-1]
    ax.annotate(f"Final: {final_rate:.2f}",
                xy=(epochs[-1], final_rate),
                xytext=(-40, 10),
                textcoords="offset points",
                fontsize=9,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {OUT_PATH}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Environment':<22} {'Final':>7} {'Best':>7} {'Best Ep':>8}")
print("="*55)
for env_name, log_file, _ in ENVS:
    log_path = os.path.join(LOG_DIR, log_file)
    epochs, rates = parse_log(log_path)
    best_idx = np.argmax(rates)
    print(f"{env_name:<22} {rates[-1]:>7.2f} {rates[best_idx]:>7.2f} {epochs[best_idx]:>8d}")
print("="*55)
