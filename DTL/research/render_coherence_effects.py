"""Render saved paired differences; no scoring, selection, or model work."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent / 'analysis/coherence_validation_20261008'
source = ROOT / 'paired_differences.json'
values = json.loads(source.read_text())
source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
labels = ['Carpark / DLinear', 'Carpark / iTransformer',
          'Finland / DLinear', 'Finland / iTransformer']
keys = ['carpark/dlinear', 'carpark/itransformer',
        'finland/dlinear', 'finland/itransformer']
fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
for stage, color, offset, name in [
    ('validation', '#1F6E8C', -.10, 'Validation development'),
    ('test_legacy', '#D17A22', .10, 'Legacy test (exploratory)'),
]:
    for i, key in enumerate(keys):
        row = values[f'{stage}/{key}']['joint_repair']['vs_original_replay']
        for ax, metric, ci in [
            (axes[0], 'accuracy_pp', 'raw_accuracy_pp_95ci'),
            (axes[1], 'mse_percent', 'mse_percent_95ci')]:
            lo, hi = row['three_day'][ci]
            val = row[metric]
            ax.errorbar(val, i+offset, xerr=np.array([[val-lo], [hi-val]]),
                        fmt='o', color=color, capsize=3,
                        label=name if i == 0 else None)
for ax in axes:
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.axvline(0, color='#888888', linewidth=.7)
    ax.grid(axis='x', alpha=.2)
axes[0].set_xlabel('Raw accuracy change vs original DP (pp)')
axes[0].xaxis.set_major_locator(MaxNLocator(5))
axes[1].set_xlabel('MSE change vs original DP (%)')
axes[0].set_title('First-step joint repair: direction')
axes[1].set_title('First-step joint repair: point error')
fig.suptitle('Fixed rules; three-day paired 95% intervals; no new model fits', fontsize=11)
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc='lower center', ncol=2, fontsize=9,
           bbox_to_anchor=(.5, .005), frameon=False)
fig.tight_layout(rect=(0, .10, 1, .97))
output = ROOT / 'coherence_fixed_repair_effects.png'
fig.savefig(output, dpi=180)
plt.close(fig)
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
(ROOT / 'plot_render_manifest.json').write_text(json.dumps({
    'scope': 'Rendering only; identical saved estimates and intervals; legend outside axes.',
    'source': str(source), 'source_sha256': source_digest,
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'output': str(output), 'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
}, indent=2))
print(output)
