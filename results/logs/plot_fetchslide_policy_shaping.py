"""
plot_fetchslide_policy_shaping.py
Generates all thesis figures for FetchSlide policy shaping results.
NOTE: violation_rate is already in % units in the log (e.g. 1.41 = 1.41%)
      success_rate, beta, shaped_rate are in [0,1] fraction units.

Usage: python plot_fetchslide_policy_shaping.py
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
BASELINE_DIR = os.path.expanduser("~/lfe_results/thesis_baseline/slide")
SHAPING_DIR  = os.path.expanduser("~/lfe_results/thesis_policy_shaping/slide")
OUT_DIR      = os.path.expanduser("~/lfe_results/thesis_plots/fetchslide")
SEEDS        = [0, 1, 2, 3, 4]
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_COLOR = "#d62728"
SHAPING_COLOR  = "#2ca02c"

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_metric(log_path, metric):
    values = []
    pat = re.compile(r'\|\s*' + re.escape(metric) + r'\s*\|\s*([0-9.e+-]+)\s*\|')
    with open(log_path, 'r') as f:
        for line in f:
            m = pat.search(line)
            if m:
                values.append(float(m.group(1)))
    return np.array(values)

def load_seeds(log_dir, metric, seeds=SEEDS):
    all_data = []
    for seed in seeds:
        path = os.path.join(log_dir, f"seed_{seed}.log")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        data = parse_metric(path, metric)
        if len(data) > 0:
            all_data.append(data)
    if not all_data:
        return None, None, None
    min_len = min(len(d) for d in all_data)
    all_data = np.array([d[:min_len] for d in all_data])
    mean   = np.mean(all_data, axis=0)
    std    = np.std(all_data, axis=0)
    stderr = std / np.sqrt(len(all_data))
    return mean, std, stderr

def plot_band(ax, epochs, mean, std, stderr, color, label, clip_max=None):
    lo_std = mean - std if clip_max is None else np.clip(mean - std, 0, clip_max)
    hi_std = mean + std if clip_max is None else np.clip(mean + std, 0, clip_max)
    lo_se  = mean - stderr if clip_max is None else np.clip(mean - stderr, 0, clip_max)
    hi_se  = mean + stderr if clip_max is None else np.clip(mean + stderr, 0, clip_max)
    ax.fill_between(epochs, lo_std, hi_std, alpha=0.12, color=color)
    ax.fill_between(epochs, lo_se,  hi_se,  alpha=0.30, color=color)
    ax.plot(epochs, mean, color=color, linewidth=2.0, label=label)

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading baseline data...")
b_succ_m, b_succ_s, b_succ_se = load_seeds(BASELINE_DIR, "test/success_rate")

print("Loading policy shaping data...")
ps_succ_m,  ps_succ_s,  ps_succ_se  = load_seeds(SHAPING_DIR, "test/success_rate")
# violation_rate is already in % units in the log
ps_viol_m,  ps_viol_s,  ps_viol_se  = load_seeds(SHAPING_DIR, "test/safety/violation_rate")
ps_beta_m,  ps_beta_s,  ps_beta_se  = load_seeds(SHAPING_DIR, "test/safety/mean_beta")
ps_shape_m, ps_shape_s, ps_shape_se = load_seeds(SHAPING_DIR, "test/safety/shaped_rate")

min_len = min(len(b_succ_m), len(ps_succ_m))
epochs  = np.arange(min_len)

# Final aggregate values (last 100 epochs)
b_succ_final  = np.mean(b_succ_m[-100:]) * 100
b_succ_std    = np.mean(b_succ_s[-100:]) * 100
ps_succ_final = np.mean(ps_succ_m[-100:]) * 100
ps_succ_std   = np.mean(ps_succ_s[-100:]) * 100
ps_viol_final = np.mean(ps_viol_m[-100:])   # already in %
ps_viol_std   = np.mean(ps_viol_s[-100:])   # already in %
ps_beta_final = np.mean(ps_beta_m[-100:])
ps_shape_final= np.mean(ps_shape_m[-100:])

print(f"Epochs: {min_len}")
print(f"Baseline final:          {b_succ_final:.2f}% ± {b_succ_std:.2f}%")
print(f"Policy shaping success:  {ps_succ_final:.2f}% ± {ps_succ_std:.2f}%")
print(f"Policy shaping violation:{ps_viol_final:.3f}% ± {ps_viol_std:.3f}%")
print(f"Mean beta:               {ps_beta_final:.3f}")
print(f"Shaped rate:             {ps_shape_final:.3f}")

# ── Figure 1 — Success Rate Comparison ────────────────────────────────────────
print("\nGenerating Figure 1 — Success Rate Comparison...")
fig, ax = plt.subplots(figsize=(9, 5))

plot_band(ax, epochs[:min_len], b_succ_m[:min_len],
          np.maximum(b_succ_s[:min_len], 0.01),
          np.maximum(b_succ_se[:min_len], 0.004),
          BASELINE_COLOR, "LFE Baseline", clip_max=1.0)

plot_band(ax, epochs[:min_len], ps_succ_m[:min_len],
          np.maximum(ps_succ_s[:min_len], 0.01),
          np.maximum(ps_succ_se[:min_len], 0.004),
          SHAPING_COLOR, "LFE + Policy Shaping", clip_max=1.0)

ax.set_title("FetchSlide-v1 — Task Success Rate\nmean (solid) ± stderr (dark) ± std (light) over 5 seeds",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=11)
ax.set_ylabel("Test Success Rate", fontsize=11)
ax.set_xlim(0, min_len-1)
ax.set_ylim(0.0, 1.05)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(fontsize=10, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.97, 0.08,
        f"Baseline: {b_succ_final:.2f}%  |  Policy Shaping: {ps_succ_final:.2f}%",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom", color="gray")
plt.tight_layout()
out = os.path.join(OUT_DIR, "fetchslide_figure1_success.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── Figure 2 — Safety Metrics Over Training ───────────────────────────────────
print("\nGenerating Figure 2 — Safety Metrics...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("FetchSlide-v1 — Policy Shaping Safety Metrics over Training\n"
             "mean ± stderr (dark) ± std (light) over 5 seeds",
             fontsize=12, fontweight="bold", y=1.02)

# Violation rate — already in % units, y-axis in %
ax = axes[0]
viol_ylim = max(4.0, np.percentile(ps_viol_m, 95) * 1.3)
plot_band(ax, epochs, ps_viol_m, ps_viol_s, ps_viol_se, SHAPING_COLOR, "Violation Rate")
ax.set_title("Safety Violation Rate", fontsize=11, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=10); ax.set_ylabel("Violation Rate (%)", fontsize=10)
ax.set_xlim(0, min_len-1); ax.set_ylim(0.0, viol_ylim)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.95, f"Final: {ps_viol_final:.3f}%",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        color=SHAPING_COLOR, fontweight="bold")

# Beta activation — fraction [0,1]
ax = axes[1]
plot_band(ax, epochs, ps_beta_m, ps_beta_s, ps_beta_se, "#ff7f0e", "Mean β")
ax.set_title("β Activation Rate", fontsize=11, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=10); ax.set_ylabel("Mean β", fontsize=10)
ax.set_xlim(0, min_len-1); ax.set_ylim(0.0, 1.0)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.95, f"Final: {ps_beta_final:.3f}",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        color="#ff7f0e", fontweight="bold")

# Shaped action rate — fraction [0,1]
ax = axes[2]
plot_band(ax, epochs, ps_shape_m, ps_shape_s, ps_shape_se, "#9467bd", "Shaped Rate")
ax.set_title("Shaped Action Rate", fontsize=11, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=10); ax.set_ylabel("Shaped Action Rate", fontsize=10)
ax.set_xlim(0, min_len-1); ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.95, f"Final: {ps_shape_final:.3f}",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        color="#9467bd", fontweight="bold")

plt.tight_layout()
out = os.path.join(OUT_DIR, "fetchslide_figure2_safety.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── Figure 3 — Safety-Performance Tradeoff ────────────────────────────────────
print("\nGenerating Figure 3 — Safety-Performance Tradeoff...")
fig, ax = plt.subplots(figsize=(9, 5))

# violation already in %, success in [0,1] → convert success to %
eps = 0.001
ratio_ps = (ps_succ_m[:min_len] * 100) / (ps_viol_m[:min_len] + eps)
ratio_ps = np.clip(ratio_ps, 0, 100)

kernel = np.ones(30) / 30
ratio_smooth = np.convolve(ratio_ps, kernel, mode='valid')
smooth_ep = epochs[14:14+len(ratio_smooth)]

ax.plot(ratio_ps, color=SHAPING_COLOR, linewidth=1.0, alpha=0.25, label="_nolegend_")
ax.plot(smooth_ep, ratio_smooth, color=SHAPING_COLOR, linewidth=2.5,
        label="LFE + Policy Shaping (smoothed)")

ax.set_title("FetchSlide-v1 — Safety-Performance Tradeoff\n"
             "Success Rate (%) / (Violation Rate (%) + ε)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=11)
ax.set_ylabel("Success / Violation Ratio", fontsize=11)
ax.set_xlim(0, min_len-1)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(fontsize=10, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
out = os.path.join(OUT_DIR, "fetchslide_figure3_tradeoff.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── Figure 4 — Baseline vs Policy Shaping Bar Chart ──────────────────────────
print("\nGenerating Figure 4 — Comparison Bar Chart...")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("FetchSlide-v1 — Baseline vs Policy Shaping Comparison",
             fontsize=13, fontweight="bold")

# Success rate
ax = axes[0]
bars = ax.bar(["LFE Baseline", "LFE + Policy Shaping"],
              [b_succ_final, ps_succ_final],
              color=[BASELINE_COLOR, SHAPING_COLOR],
              yerr=[b_succ_std, ps_succ_std],
              capsize=5, alpha=0.85)
ax.set_title("Task Success Rate", fontsize=11, fontweight="bold")
ax.set_ylabel("Success Rate (%)", fontsize=10)
ax.set_ylim(0, 65)
ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
ax.grid(True, alpha=0.3, linestyle="--", axis='y')
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for bar, val in zip(bars, [b_succ_final, ps_succ_final]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

# Violation rate — already in %
ax = axes[1]
viol_ylim = max(5.0, ps_viol_final * 1.5 + ps_viol_std * 1.5)
bars2 = ax.bar(["LFE Baseline", "LFE + Policy Shaping"],
               [0, ps_viol_final],
               color=[BASELINE_COLOR, SHAPING_COLOR],
               yerr=[0, ps_viol_std],
               capsize=5, alpha=0.85)
ax.set_title("Safety Constraint Violation Rate", fontsize=11, fontweight="bold")
ax.set_ylabel("Violation Rate (%)", fontsize=10)
ax.set_ylim(0, viol_ylim)
ax.grid(True, alpha=0.3, linestyle="--", axis='y')
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0, viol_ylim * 0.03, "N/A", ha='center', va='bottom', fontsize=10, color='gray')
ax.text(1, ps_viol_final + ps_viol_std * 0.1 + viol_ylim * 0.02,
        f"{ps_viol_final:.3f}%", ha='center', va='bottom',
        fontsize=10, fontweight='bold', color=SHAPING_COLOR)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fetchslide_figure4_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── Figure 5 — Per-Seed Results ───────────────────────────────────────────────
print("\nGenerating Figure 5 — Per-Seed Results...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("FetchSlide-v1 — Per-Seed Policy Shaping Results (last 100 epochs)",
             fontsize=12, fontweight="bold")

seed_success = []
seed_violation = []
for seed in SEEDS:
    path = os.path.join(SHAPING_DIR, f"seed_{seed}.log")
    if os.path.exists(path):
        s = parse_metric(path, "test/success_rate")
        v = parse_metric(path, "test/safety/violation_rate")
        seed_success.append(np.mean(s[-100:]) * 100 if len(s) >= 100 else np.mean(s) * 100)
        # violation already in %
        seed_violation.append(np.mean(v[-100:]) if len(v) >= 100 else np.mean(v))

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
x = np.arange(len(SEEDS))

# Success rate per seed
ax = axes[0]
bars = ax.bar(x, seed_success, color=colors, alpha=0.85)
ax.axhline(np.mean(seed_success), color='black', linewidth=1.5,
           linestyle='--', label=f"Mean: {np.mean(seed_success):.2f}%")
ax.set_title("Success Rate per Seed", fontsize=11, fontweight="bold")
ax.set_xlabel("Seed", fontsize=10); ax.set_ylabel("Success Rate (%)", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([f"Seed {s}" for s in SEEDS])
ax.set_ylim(0, 45)
ax.set_yticks([0, 10, 20, 30, 40])
ax.grid(True, alpha=0.3, linestyle="--", axis='y')
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for bar, val in zip(bars, seed_success):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha='center', va='bottom', fontsize=9)

# Violation rate per seed — already in %
ax = axes[1]
viol_ylim = max(4.0, max(seed_violation) * 1.4)
bars = ax.bar(x, seed_violation, color=colors, alpha=0.85)
ax.axhline(np.mean(seed_violation), color='black', linewidth=1.5,
           linestyle='--', label=f"Mean: {np.mean(seed_violation):.3f}%")
ax.set_title("Violation Rate per Seed", fontsize=11, fontweight="bold")
ax.set_xlabel("Seed", fontsize=10); ax.set_ylabel("Violation Rate (%)", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([f"Seed {s}" for s in SEEDS])
ax.set_ylim(0, viol_ylim)
ax.grid(True, alpha=0.3, linestyle="--", axis='y')
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for bar, val in zip(bars, seed_violation):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + viol_ylim * 0.01,
            f"{val:.3f}%", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fetchslide_figure5_per_seed.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── Table 1 Data ──────────────────────────────────────────────────────────────
print("\nGenerating Table 1 data...")
out = os.path.join(OUT_DIR, "fetchslide_table1_data.txt")
with open(out, 'w') as f:
    f.write("="*70 + "\n")
    f.write("Table 1 — FetchSlide-v1 Aggregate Results (last 100 epochs)\n")
    f.write("="*70 + "\n")
    f.write(f"{'Method':<25} {'Success Rate':>14} {'Violation Rate':>16} {'Mean β':>10} {'Shaped Rate':>13}\n")
    f.write("-"*70 + "\n")
    f.write(f"{'LFE Baseline':<25} {b_succ_final:>11.2f}%±{b_succ_std:.2f}% {'N/A':>16} {'N/A':>10} {'N/A':>13}\n")
    f.write(f"{'LFE + Policy Shaping':<25} {ps_succ_final:>11.2f}%±{ps_succ_std:.2f}% {ps_viol_final:>13.3f}%±{ps_viol_std:.3f}% {ps_beta_final:>10.3f} {ps_shape_final:>13.3f}\n")
    f.write("="*70 + "\n")
    f.write(f"\nSeeds: 5 | Epochs: 1000 | Workers: 19 MPI\n")

with open(out, 'r') as f:
    print(f.read())

print(f"\nAll figures saved to: {OUT_DIR}")
print("Copy to GitHub repo with:")
print(f"  cp {OUT_DIR}/*.png ~/adaptive-safe-rl-policy-shaping/results/fetchslide/")
print(f"  cp {OUT_DIR}/*.txt ~/adaptive-safe-rl-policy-shaping/results/fetchslide/")
