"""
plot_pickandplace_baseline.py
Generates thesis figures for FetchPickAndPlace baseline results.
Figures generated:
  1. fetchpickandplace_baseline.png  — 5-seed learning curve
  2. fetchpickandplace_table_baseline.txt — aggregate results

Usage: python plot_pickandplace_baseline.py
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
BASELINE_DIR = os.path.expanduser("~/lfe_results/thesis_baseline/pickandplace")
OUT_DIR      = os.path.expanduser("~/lfe_results/thesis_plots")
SEEDS        = [0, 1, 2, 3, 4]
os.makedirs(OUT_DIR, exist_ok=True)

LFE_COLOR = "#d62728"  # red — matches Mingkang's Figure 1 style

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_log(path):
    epochs, rates = [], []
    ep  = re.compile(r'\|\s*epoch\s*\|\s*(\d+)\s*\|')
    rp  = re.compile(r'\|\s*test/success_rate\s*\|\s*([0-9.e+-]+)\s*\|')
    cur = None
    with open(path, 'r') as f:
        for line in f:
            em = ep.search(line)
            if em: cur = int(em.group(1))
            rm = rp.search(line)
            if rm and cur is not None:
                epochs.append(cur)
                rates.append(float(rm.group(1)))
                cur = None
    return np.array(epochs), np.array(rates)

# ── Load 5-seed data ──────────────────────────────────────────────────────────
all_rates, available_seeds = [], []
for seed in SEEDS:
    lp = os.path.join(BASELINE_DIR, f"seed_{seed}.log")
    if not os.path.exists(lp):
        print(f"Seed {seed}: not found — skipping")
        continue
    _, rates = parse_log(lp)
    all_rates.append(rates)
    available_seeds.append(seed)
    print(f"Seed {seed}: {len(rates)} epochs, final={rates[-1]:.3f}")

min_len   = min(len(r) for r in all_rates)
all_rates = np.array([r[:min_len] for r in all_rates])
epochs    = np.arange(min_len)
mean      = np.mean(all_rates, axis=0)
std       = np.std(all_rates, axis=0)
stderr    = std / np.sqrt(len(available_seeds))

cm = np.mean(mean[-100:])
cs = np.mean(std[-100:])
print(f"\nFinal: {cm*100:.2f}% ± {cs*100:.2f}%")

# Find convergence epoch (first epoch where mean > 0.95)
conv_epochs = np.where(mean > 0.95)[0]
conv_epoch  = conv_epochs[0] if len(conv_epochs) > 0 else -1
print(f"Convergence epoch (>95%): {conv_epoch}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.fill_between(epochs,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                alpha=0.15, color=LFE_COLOR)
ax.fill_between(epochs,
                np.clip(mean - stderr, 0, 1),
                np.clip(mean + stderr, 0, 1),
                alpha=0.35, color=LFE_COLOR)
ax.plot(epochs, mean, color=LFE_COLOR, linewidth=2.0, label="LFE")

ax.set_title("FetchPickAndPlace-v1 — LFE Baseline\n"
             "mean (solid) ± stderr (dark) ± std (light) over 5 seeds",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=11)
ax.set_ylabel("Test Success Rate", fontsize=11)
ax.set_xlim(0, min_len - 1)
ax.set_ylim(0.0, 1.05)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(fontsize=10, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.97, 0.08,
        f"Final: {cm*100:.2f}% ± {cs*100:.2f}%",
        transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
        color=LFE_COLOR, fontweight="bold")

plt.tight_layout()
out = os.path.join(OUT_DIR, "fetchpickandplace_baseline.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── Table data ────────────────────────────────────────────────────────────────
out_txt = os.path.join(OUT_DIR, "fetchpickandplace_table_baseline.txt")
with open(out_txt, 'w') as f:
    f.write("="*60 + "\n")
    f.write("FetchPickAndPlace-v1 — Baseline Results\n")
    f.write("="*60 + "\n")
    f.write(f"{'Method':<25} {'Success Rate':>15} {'Convergence':>12}\n")
    f.write("-"*60 + "\n")
    f.write(f"{'LFE Baseline':<25} {cm*100:>12.2f}%±{cs*100:.2f}% {conv_epoch:>12}\n")
    f.write("="*60 + "\n")
    f.write(f"\nSeeds: 5 | Epochs: 1000 | Workers: 19 MPI\n")
    f.write("\nPer-seed final success rates:\n")
    for i, seed in enumerate(available_seeds):
        f.write(f"  Seed {seed}: {all_rates[i][-1]*100:.1f}%\n")

with open(out_txt, 'r') as f:
    print(f.read())

print(f"\nAll files saved to: {OUT_DIR}")
print("Copy to GitHub repo with:")
print(f"cp {OUT_DIR}/fetchpickandplace*.png ~/adaptive-safe-rl-policy-shaping/results/")
