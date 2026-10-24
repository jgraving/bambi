"""Generate a comparison figure for the logexpm1 pull request."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit

eta = np.linspace(-8, 8, 801)
exp_value = np.exp(eta)
softplus_value = np.logaddexp(0, eta)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
colors = {'exp': '#3267A8', 'softplus': '#D96B27'}
for ax, (title, exponential, softplus) in zip(axes, [
    ('Inverse-link magnitude', exp_value, softplus_value),
    ('Inverse-link derivative', exp_value, expit(eta)),
]):
    ax.plot(eta, exponential, color=colors['exp'], lw=2.2, label=r'$\exp(\eta)$')
    ax.plot(eta, softplus, color=colors['softplus'], lw=2.2, label='softplus' if ax is axes[0] else 'sigmoid')
    ax.set_yscale('log')
    ax.set(xlim=(-8,8), ylim=(1e-4,1e4), xlabel=r'Linear predictor $\eta$', title=title)
    ax.set_ylabel('Value (log scale)' if ax is axes[0] else 'Derivative (log scale)')
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.axhline(1, color='#777777', lw=.7, ls=':', zorder=0)
    ax.legend(frameon=False, loc='upper left')
fig.savefig(Path(__file__).with_name('logexpm1-comparison.png'), dpi=180, facecolor='white')
