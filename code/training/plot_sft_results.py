"""Reproduce paper-style SFT figures from verified Trainer and probe records."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def panel(ax, letter):
    ax.text(-.02, 1.035, f'({letter})', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=9, fontweight='bold')


def render(source, output):
    data = json.loads(Path(source).read_text(encoding='utf-8'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':['Times New Roman', 'DejaVu Serif'],
        'font.size':8.5, 'axes.titlesize':10, 'axes.labelsize':8.5,
        'xtick.labelsize':8, 'ytick.labelsize':8,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.linewidth':.65, 'xtick.major.width':.6, 'ytick.major.width':.6,
        'xtick.major.size':3, 'ytick.major.size':3,
        'figure.facecolor':'white', 'savefig.facecolor':'white',
        'axes.grid':True, 'axes.axisbelow':True,
        'grid.color':'#D8D8D8', 'grid.alpha':.6, 'grid.linewidth':.4,
    })
    keys = ['streetview', 'satellite']
    colors = {'streetview':'#2166AC', 'satellite':'#C76C18'}
    names = {'streetview':'StreetView', 'satellite':'Satellite'}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.05))
    fig.subplots_adjust(left=.145, right=.99, top=.88, bottom=.115,
                        wspace=.48, hspace=.42)
    for row, key in enumerate(keys):
        d = data[key]
        train = d['training']
        xx = [r['step'] for r in train]
        color = colors[key]
        ax = axes[row, 0]
        ax.plot(xx, [r['loss'] for r in train], color=color, lw=1.35, label='Train')
        ax.plot([r['step'] for r in d['validation']], [r['loss'] for r in d['validation']],
                color='#555555', ls='--', marker='o', ms=2.8, lw=1.0, label='Validation')
        ax.set(ylabel='Cross-entropy', ylim=(.7, 1.9))
        ax.set_yticks([.8, 1.1, 1.4, 1.7])
        if row == 0:
            ax.legend(loc='upper right', frameon=True, facecolor='white',
                      edgecolor='none', framealpha=.95, fontsize=7.5,
                      handlelength=1.7, handletextpad=.5, labelspacing=.25)
        axes[row, 1].plot(xx, [r['learning_rate']*1e5 for r in train], color=color, lw=1.35)
        axes[row, 1].set(ylabel=r'LR ($\times10^{-5}$)', ylim=(-.1, 5.3))
        axes[row, 1].set_yticks([0, 2, 4])
        axes[row, 2].plot(xx, [r['grad_norm'] for r in train], color=color, lw=1.1)
        axes[row, 2].set(ylabel='Grad. norm', ylim=(0, 1.1))
        axes[row, 2].set_yticks([0, .25, .5, .75, 1])
        for col in range(3):
            ax = axes[row, col]
            ax.set_xlim(0, 820)
            ax.set_xticks([0, 200, 400, 600, 800])
            ax.axvline(data['selection'][key], color=color, ls='--', lw=.85, alpha=.85)
            panel(ax, 'abcdef'[row*3+col])
            if row == 1:
                ax.set_xlabel('Optimizer update')
        pos = axes[row, 0].get_position()
        fig.text(.027, (pos.y0+pos.y1)/2, names[key], rotation=90,
                 ha='center', va='center', fontsize=10, fontweight='bold', color=color)
    for col, title in enumerate(['Loss', 'Learning rate', 'Gradient norm']):
        axes[0, col].set_title(title, pad=18)
    fig.savefig(output/'training_dynamics.png', dpi=400)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    fig.subplots_adjust(left=.085, right=.99, top=.77, bottom=.185, wspace=.28)
    for col, (ax, field, label) in enumerate(zip(
            axes, ['em', 'adjacency'], ['Exact match (%)', 'Directed adjacency (%)'])):
        for key in keys:
            pp = data[key]['probes']
            ax.plot([p['step'] for p in pp], [p[field]*100 for p in pp], 'o-',
                    color=colors[key], lw=1.35, ms=3.7, markeredgewidth=.5,
                    markeredgecolor='white', label=names[key])
            chosen = next(p for p in pp if p['step'] == data['selection'][key])
            ax.scatter(chosen['step'], chosen[field]*100, marker='*', s=72,
                       color=colors[key], edgecolors='white', linewidth=.6, zorder=6)
        ax.set(xlabel='Optimizer update', ylabel=label, xlim=(-20, 820), ylim=(-3, 100))
        ax.set_xticks([0, 200, 400, 600, 800])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        panel(ax, 'ab'[col])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, .985),
               ncol=2, frameon=False, handlelength=2.2, columnspacing=2.0)
    fig.savefig(output/'checkpoint_validation_dynamics_with_baseline.png', dpi=400)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    render(a.source, a.output)
