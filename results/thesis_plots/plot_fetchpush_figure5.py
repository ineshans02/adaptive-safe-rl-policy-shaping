import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 150, 'font.size': 11, 'axes.titlesize': 13,
    'axes.titleweight': 'bold', 'axes.labelsize': 11, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 10})

BLUE = '#4C72B0'; ORANGE = '#DD8452'; RED = '#C44E52'; GREEN = '#55A868'

seeds = [0,1,2,3,4]
ps_success   = np.array([1.000, 0.979, 0.995, 1.000, 1.000])
ps_violation = np.array([0.00474, 0.00421, 0.000, 0.000, 0.00474])
bl_success   = np.array([0.9995, 0.9979, 0.9984, 0.9995, 0.9979])

ps_s_m = ps_success.mean(); ps_v_m = ps_violation.mean(); bl_s_m = bl_success.mean()
seed_labels = [f'Seed {i}' for i in seeds]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
bars = ax.bar(seed_labels, ps_success*100, color=ORANGE, alpha=0.85, edgecolor='white')
ax.axhline(y=bl_s_m*100, color=BLUE, linestyle='--', linewidth=1.5, label=f'Baseline mean ({bl_s_m*100:.2f}%)')
ax.axhline(y=ps_s_m*100, color=RED, linestyle='--', linewidth=1.5, label=f'PS mean ({ps_s_m*100:.2f}%)')
ax.set_ylabel('Success Rate (%)'); ax.set_title('Final Success Rate per Seed')
ax.set_ylim(94, 103); ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3, axis='y')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

ax = axes[1]
bar_colors = [GREEN if v==0 else ORANGE if v<0.01 else RED for v in ps_violation]
bars = ax.bar(seed_labels, ps_violation*100, color=bar_colors, alpha=0.85, edgecolor='white')
ax.axhline(y=ps_v_m*100, color='black', linestyle='--', linewidth=1.5, label=f'Mean ({ps_v_m*100:.3f}%)')
ax.set_ylabel('Violation Rate (%)'); ax.set_title('Final Violation Rate per Seed')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, ps_violation):
    label = f'{val*100:.3f}%' if val > 0 else '0%'
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            label, ha='center', va='bottom', fontsize=9)

plt.suptitle('FetchPush Policy Shaping — Per-Seed Results', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ineshans/lfe_results/thesis_plots/fetchpush/fetchpush_figure5_per_seed.png', dpi=150, bbox_inches='tight')
print("Done: fetchpush_figure5_per_seed.png")
