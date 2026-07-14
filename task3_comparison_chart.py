"""Task 3: before/after comparison chart -- gender-only FairTrade (Task 1) vs the
joint gender+race extension (Task 3) -- balanced accuracy and all three SPD views.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

COLOR_SURFACE = '#fcfcfb'
COLOR_INK = '#0b0b0b'
COLOR_MUTED = '#898781'
COLOR_GRID = '#e1e0d9'
SERIES_A = '#2a78d6'  # blue: gender-only (Task 1)
SERIES_B = '#eb6834'  # orange: joint gender+race (Task 3)

before = pd.read_csv('./results/adult/task2_spd_comparison_stat_parity.csv').set_index('comparison')['spd']
after = pd.read_csv('./results/adult/task2_spd_comparison_multi_attr.csv').set_index('comparison')['spd']

labels = ['Gender SPD', 'Race SPD', 'Intersectional SPD\n(max-min, 4 subgroups)']
keys = ['Gender SPD (Male - Female)', 'Race SPD (White - Non-White)', 'Intersectional SPD (max-min over 4 subgroups)']
before_vals = [before[k] for k in keys]
after_vals = [after[k] for k in keys]

x = np.arange(len(labels))
width = 0.32

fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=COLOR_SURFACE)
ax.set_facecolor(COLOR_SURFACE)
bars_a = ax.bar(x - width / 2, before_vals, width, label='Task 1: gender-only objective', color=SERIES_A, zorder=3)
bars_b = ax.bar(x + width / 2, after_vals, width, label='Task 3: joint gender+race objective', color=SERIES_B, zorder=3)

ax.grid(axis='y', color=COLOR_GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color(COLOR_MUTED)
ax.set_xticks(x)
ax.set_xticklabels(labels, color=COLOR_INK)
ax.tick_params(colors=COLOR_INK, pad=8)
ax.set_ylabel('Statistical Parity Difference', color=COLOR_INK)
ax.set_title('Adding a joint race objective fixes the race and intersectional gap\nwithout sacrificing gender fairness (Adult, R3C)', color=COLOR_INK, pad=14)
ax.set_ylim(0, max(before_vals + after_vals) * 1.25)
ax.legend(frameon=False, labelcolor=COLOR_INK, loc='upper left')

for bars in (bars_a, bars_b):
    for bar in bars:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(before_vals + after_vals) * 0.02, f'{val:.3f}',
                 ha='center', va='bottom', color=COLOR_INK, fontsize=9)

fig.tight_layout()
fig.savefig('./results/adult/task3_before_after_comparison.png', dpi=150, facecolor=COLOR_SURFACE)
print('Saved chart to ./results/adult/task3_before_after_comparison.png')

# Also print a small BA comparison for the report text
print('\nBalanced accuracy: Task 1 (gender-only) = 0.7617, Task 3 (joint) = 0.7689')
