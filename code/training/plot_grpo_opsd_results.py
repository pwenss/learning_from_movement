"""Reproduce the two paper-style GRPO+OPSD figures from frozen audit data."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def trailing(values, window):
    a = np.asarray(values, dtype=float)
    return np.asarray([np.mean(a[max(0, i-window+1):i+1]) for i in range(len(a))])


def panel(ax, letter):
    ax.text(-.02, 1.035, f'({letter})', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=9, fontweight='bold')


def render(source, output):
    data = json.loads(Path(source).read_text(encoding='utf-8'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8.5, 'axes.titlesize': 10, 'axes.labelsize': 8.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.grid': True, 'axes.axisbelow': True,
        'grid.color': '#D8D8D8', 'grid.alpha': .6, 'grid.linewidth': .4,
    })
    colors = {'streetview': '#2166AC', 'satellite': '#C76C18'}
    names = {'streetview': 'StreetView', 'satellite': 'Satellite'}
    keys = ['streetview', 'satellite']
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.05))
    fig.subplots_adjust(left=.145, right=.99, top=.88, bottom=.115,
                        wspace=.48, hspace=.42)
    for row, key in enumerate(keys):
        d = data[key]
        color = colors[key]
        selected = data['selection'][key]
        complete = [r for r in d['rollouts'] if key != 'satellite' or r['cycle'] <= 69]
        x = [r['cycle'] for r in complete]
        y = [r['em']*100 for r in complete]
        ax = axes[row, 0]
        ax.plot(x, y, color=color, alpha=.24, lw=.65)
        ax.plot(x, trailing(y, 5), color=color, lw=1.35)
        ax.set(ylabel='EM (%)', ylim=(0, 100), xlim=(0, 72))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks([0, 20, 40, 60])
        if key == 'satellite':
            last = d['rollouts'][-1]
            ax.scatter(last['cycle'], last['em']*100, facecolors='none',
                       edgecolors=color, s=16, linewidths=.8, zorder=5)
        if key == 'streetview':
            ax.axvline(51, color='#888888', ls=':', lw=.75)
        ax.axvline(selected, color=color, ls='--', lw=.85, alpha=.85)
        updates = d['updates']
        ux = [u['update'] for u in updates]
        for col, field, label in [(1, 'entropy_mc', 'Nats / token'),
                                  (2, 'reference_kl', r'KL ($\times10^{-3}$)')]:
            ax = axes[row, col]
            values = np.asarray([u[field] for u in updates])
            assert np.isfinite(values).all()
            if field == 'reference_kl':
                values = values*1000  # Display unit only; not the loss coefficient.
            ax.plot(ux, values, color=color, alpha=.24, lw=.55)
            ax.plot(ux, trailing(values, 8), color=color, lw=1.3)
            ax.axvline(selected*4, color=color, ls='--', lw=.85, alpha=.85)
            ax.set(ylabel=label, xlim=(0, 288))
            ax.set_xticks([0, 100, 200])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        for col in range(3):
            ax = axes[row, col]
            panel(ax, 'abcdef'[row*3+col])
            if row == 1:
                ax.set_xlabel('Rollout cycle' if col == 0 else 'Optimizer update')
    for col, title in enumerate(['Rollout EM', 'Entropy (MC)', 'Reference KL']):
        axes[0, col].set_title(title, pad=18)
    for row, key in enumerate(keys):
        pos = axes[row, 0].get_position()
        fig.text(.027, (pos.y0+pos.y1)/2, names[key], rotation=90,
                 ha='center', va='center', fontsize=10, fontweight='bold', color=colors[key])
    fig.savefig(output/'training_dynamics.png', dpi=400)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    fig.subplots_adjust(left=.085, right=.99, top=.77, bottom=.185, wspace=.28)
    for col, (ax, field, label) in enumerate(zip(
            axes, ['em', 'adjacency'], ['Exact match (%)', 'Directed adjacency (%)'])):
        for key in keys:
            probes = data[key]['probes']
            xx = [p['update'] for p in probes]
            yy = [p[field]*100 for p in probes]
            ax.plot(xx, yy, 'o-', color=colors[key], lw=1.35, ms=3.7,
                    markeredgewidth=.5, markeredgecolor='white', label=names[key])
            chosen = next(p for p in probes if p['cycle'] == data['selection'][key])
            ax.scatter(chosen['update'], chosen[field]*100, marker='*', s=72,
                       color=colors[key], edgecolors='white', linewidth=.6, zorder=6)
        ax.set(xlabel='Optimizer update', ylabel=label, xlim=(-8, 283),
               ylim=(59, 95) if field == 'em' else (79, 97))
        ax.set_xticks([0, 68, 136, 204, 272])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        panel(ax, 'ab'[col])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, .985),
               ncol=2, frameon=False, handlelength=2.2, columnspacing=2.0)
    fig.savefig(output/'checkpoint_probe_dynamics.png', dpi=400)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    render(a.source, a.output)
