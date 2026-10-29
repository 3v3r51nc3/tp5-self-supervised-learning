"""bar chart of results/results_all.csv -> report/figures/comparison.png"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(root / 'results' / 'results_all.csv')))
tasks = ['context_encoder', 'rotation', 'simclr', 'patch_position']
labels = ['Context Encoder', 'Rotation', 'SimCLR', 'Patch position']

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
for ax, ds, title in zip(axes, ['cifar10', 'cifar100'], ['CIFAR10', 'CIFAR100 (3 époques)']):
    vals = [float(next(r for r in rows if r['task'] == t and r['dataset'] == ds)['test_top1']) for t in tasks]
    bars = ax.bar(labels, vals, color=['#8c8c8c', '#4c78a8', '#59a14f', '#e15759'])
    ax.bar_label(bars, fmt='%.1f', fontsize=8)
    ax.set_title(title)
    ax.set_ylabel('Top-1 test (%)')
    ax.tick_params(axis='x', rotation=20)
fig.tight_layout()
fig.savefig(root / 'report' / 'figures' / 'comparison.png', dpi=200)
