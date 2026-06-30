"""
plot_policy_shaping_results.py
Generates all thesis figures for FetchPush policy shaping results.
Figures generated:
  1. fetchpush_figure1_comparison.png  — success rate: baseline vs policy shaping
  2. fetchpush_figure2_safety.png      — violation rate + beta + shaped rate over training
  3. fetchpush_figure3_tradeoff.png    — safety-performance tradeoff curve
  4. fetchpush_table1_data.txt         — aggregate results table data

Usage: python plot_policy_shaping_results.py
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
BASELINE_DIR  = os.path.expanduser("~/lfe_results/thesis_baseline/push")
SHAPING_DIR   = os.path.expanduser("~/lfe_results/thesis_policy_shaping/push")
OUT_DIR       = os.path.expanduser("~/lfe_results/thesis_plots")
SEEDS         = [0, 1, 2, 3, 4]
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_COLOR = "#d62728"   # red
SHAPING_COLOR  = "#2ca02c"   # green

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

def plot_band(ax, epochs, mean, std, stderr, color, label):
    ax.fill_between(epochs,
                    np.clip(mean - std, 0, 1),
                    np.clip(mean + std, 0, 1),
                    alpha=0.12, color=color)
    ax.fill_between(epochs,
                    np.clip(mean - stderr, 0, 1),
                    np.clip(mean + stderr, 0, 1),
                    alpha=0.30, color=color)
    ax.plot(epochs, mean, color=color, linewidth=2.0, label=label)

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading baseline data...")
b_succ_m, b_succ_s, b_succ_se = load_seeds(BASELINE_DIR, "test/success_rate")

print("Loading policy shaping data...")
ps_succ_m,  ps_succ_s,  ps_succ_se  = load_seeds(SHAPING_DIR, "test/success_rate")
ps_viol_m,  ps_viol_s,  ps_viol_se  = load_seeds(SHAPING_DIR, "test/safety/violation_rate")
ps_beta_m,  ps_beta_s,  ps_beta_se  = load_seeds(SHAPING_DIR, "test/safety/mean_beta")
ps_shape_m, ps_shape_s, ps_shape_se = load_seeds(SHAPING_DIR, "test/safety/shaped_rate")

min_len = min(len(b_succ_m), len(ps_succ_m))
epochs  = np.arange(min_len)

print(f"Epochs: {min_len}")
print(f"Baseline final: {np.mean(b_succ_m[-100:])*100:.2f}% ± {np.mean(b_succ_s[-100:])*100:.2f}%")
print(f"Policy shaping final success: {np.mean(ps_succ_m[-100:])*100:.2f}% ± {np.mean(ps_succ_s[-100:])*100:.2f}%")
print(f"Policy shaping final violation: {np.mean(ps_viol_m[-100:])*100:.3f}% ± {np.mean(ps_viol_s[-100:])*100:.3f}%")

# ── Figure 1 — Success Rate Comparison ────────────────────────────────────────
print("\nGenerating Figure 1 — Success Rate Comparison...")
fig, ax = plt.subplots(figsize=(9, 5))

plot_band(ax, epochs[:min_len], b_succ_m[:min_len],
          np.maximum(b_succ_s[:min_len], 0.01),
          np.maximum(b_succ_se[:min_len], 0.004),
          BASELINE_COLOR, "LFE Baseline")

plot_band(ax, epochs[:min_len], ps_succ_m[:min_len],
          np.maximum(ps_succ_s[:min_len], 0.01),
          np.maximum(ps_succ_se[:min_len], 0.004),
          SHAPING_COLOR, "LFE + Policy Shaping")

ax.set_title("FetchPush-v1 — Task Success Rate\nmean (solid) ± stderr (dark) ± std (light) over 5 seeds",
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
b_final = np.mean(b_succ_m[-100:])*100
ps_final = np.mean(ps_succ_m[-100:])*100
ax.text(0.97, 0.08,
        f"Baseline: {b_final:.2f}%  |  Policy Shaping: {ps_final:.2f}%",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        color="gray")
plt.tight_layout()
out = os.path.join(OUT_DIR, "fetchpush_figure1_success.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── Figure 2 — Safety Metrics Over Training ───────────────────────────────────
print("\nGenerating Figure 2 — Safety Metrics...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("FetchPush-v1 — Policy Shaping Safety Metrics over Training\n"
             "mean ± stderr (dark) ± std (light) over 5 seeds",
             fontsize=12, fontweight="bold", y=1.02)

# Violation rate
ax = axes[0]
plot_band(ax, epochs, ps_viol_m, ps_viol_s, ps_viol_se, SHAPING_COLOR, "Violation Rate")
ax.set_title("Safety Violation Rate", fontsize=11, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=10); ax.set_ylabel("Violation Rate", fontsize=10)
ax.set_xlim(0, min_len-1); ax.set_ylim(0.0, 0.15)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.95, f"Final: {np.mean(ps_viol_m[-100:])*100:.3f}%",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        color=SHAPING_COLOR, fontweight="bold")

# Beta activation
ax = axes[1]
plot_band(ax, epochs, ps_beta_m, ps_beta_s, ps_beta_se, "#ff7f0e", "Mean β")
ax.set_title("β Activation Rate", fontsize=11, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=10); ax.set_ylabel("Mean β", fontsize=10)
ax.set_xlim(0, min_len-1); ax.set_ylim(0.0, 0.5)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.95, f"Final: {np.mean(ps_beta_m[-100:]):.3f}",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        color="#ff7f0e", fontweight="bold")

# Shaped action rate
ax = axes[2]
plot_band(ax, epochs, ps_shape_m, ps_shape_s, ps_shape_se, "#9467bd", "Shaped Rate")
ax.set_title("Shaped Action Rate", fontsize=11, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=10); ax.set_ylabel("Shaped Action Rate", fontsize=10)
ax.set_xlim(0, min_len-1); ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.95, f"Final: {np.mean(ps_shape_m[-100:]):.3f}",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        color="#9467bd", fontweight="bold")

plt.tight_layout()
out = os.path.join(OUT_DIR, "fetchpush_figure2_safety.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── Figure 3 — Safety-Performance Tradeoff ────────────────────────────────────
print("\nGenerating Figure 3 — Safety-Performance Tradeoff...")
fig, ax = plt.subplots(figsize=(9, 5))

# Ratio = success / (violation + epsilon) — Recovery RL style metric
eps = 0.001
ratio_ps = ps_succ_m[:min_len] / (ps_viol_m[:min_len] + eps)
ratio_ps = np.clip(ratio_ps, 0, 500)

# Smooth for readability
kernel = np.ones(30) / 30
ratio_smooth = np.convolve(ratio_ps, kernel, mode='valid')
smooth_ep = epochs[14:14+len(ratio_smooth)]

ax.plot(ratio_ps, color=SHAPING_COLOR, linewidth=1.0, alpha=0.25, label="_nolegend_")
ax.plot(smooth_ep, ratio_smooth, color=SHAPING_COLOR, linewidth=2.5,
        label="LFE + Policy Shaping (smoothed)")

ax.set_title("FetchPush-v1 — Safety-Performance Tradeoff\n"
             "Success Rate / (Violation Rate + ε)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=11)
ax.set_ylabel("Success / Violation Ratio", fontsize=11)
ax.set_xlim(0, min_len-1)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(fontsize=10, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
out = os.path.join(OUT_DIR, "fetchpush_figure3_tradeoff.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ── Table 1 Data ──────────────────────────────────────────────────────────────
print("\nGenerating Table 1 data...")
out = os.path.join(OUT_DIR, "fetchpush_table1_data.txt")
with open(out, 'w') as f:
    f.write("="*70 + "\n")
    f.write("Table 1 — FetchPush-v1 Aggregate Results (last 100 epochs)\n")
    f.write("="*70 + "\n")
    f.write(f"{'Method':<25} {'Success Rate':>14} {'Violation Rate':>16} {'Mean β':>10} {'Shaped Rate':>13}\n")
    f.write("-"*70 + "\n")
    b_sm  = np.mean(b_succ_m[-100:]); b_ss = np.mean(b_succ_s[-100:])
    ps_sm = np.mean(ps_succ_m[-100:]); ps_ss = np.mean(ps_succ_s[-100:])
    ps_vm = np.mean(ps_viol_m[-100:]); ps_vs = np.mean(ps_viol_s[-100:])
    ps_bm = np.mean(ps_beta_m[-100:])
    ps_hm = np.mean(ps_shape_m[-100:])
    f.write(f"{'LFE Baseline':<25} {b_sm*100:>11.2f}%±{b_ss*100:.2f}% {'N/A':>16} {'N/A':>10} {'N/A':>13}\n")
    f.write(f"{'LFE + Policy Shaping':<25} {ps_sm*100:>11.2f}%±{ps_ss*100:.2f}% {ps_vm*100:>13.3f}%±{ps_vs*100:.3f}% {ps_bm:>10.3f} {ps_hm:>13.3f}\n")
    f.write("="*70 + "\n")
    f.write(f"\nSeeds: 5 | Epochs: 1000 | Workers: 19 MPI\n")

with open(out, 'r') as f:
    print(f.read())

print(f"\nAll figures saved to: {OUT_DIR}")
print("Copy to GitHub repo with:")
print(f"cp {OUT_DIR}/*.png ~/adaptive-safe-rl-policy-shaping/results/")
