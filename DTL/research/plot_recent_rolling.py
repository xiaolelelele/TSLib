"""Standalone scientific figure from checked CSV summaries (not a UI visual)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    args = parser.parse_args()
    root = Path(args.analysis)
    main = pd.read_csv(root/"summary/summary.csv")
    clip = pd.read_csv(root/"summary/physical_ranges_posthoc.csv")
    clip = clip[clip.method=="base_clipped"].groupby(["dataset", "year", "backbone"]).mean(numeric_only=True).reset_index()
    control = pd.read_csv(root/"controls/summary.csv")
    groups = sorted(set(tuple(row) for row in main[["dataset", "year", "backbone"]].to_numpy()))
    series = [(main[main.method=="raw_selected"], "Raw direction correction", "#176dba", "o"),
              (main[main.method=="paired_gate"], "Paired ridge gate", "#b54821", "s"),
              (clip, "Physical ranges (post hoc)", "#15805d", "^"),
              (control[control.method=="confidence_only_flat"], "Confidence-only flat (post hoc)", "#9561a9", "D")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.5), sharey=True)
    labels = [f"{'Energy' if d=='smard_de' else 'Weather'} {y} / {b}" for d,y,b in groups]
    for j, (frame, label, color, marker) in enumerate(series):
        for i, (dataset, year, backbone) in enumerate(groups):
            row = frame[(frame.dataset==dataset)&(frame.year==year)&(frame.backbone==backbone)].iloc[0]
            y = i+(j-1.5)*.13
            axes[0].scatter(row.raw_gain_pp, y, c=color, marker=marker, s=38, label=label if i==0 else None)
            axes[1].scatter(row.mse_change_pct, y, c=color, marker=marker, s=38)
    axes[0].set_yticks(range(len(groups)), labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axvline(0, color="black", lw=.8)
        ax.grid(axis="x", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("Raw sign gain vs base (percentage points; higher is better)")
    axes[1].set_xlabel("MSE change vs base (%; lower is better)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.07, -.16), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("2023–2025 recent hourly data: fixed fits and separate mechanism controls", fontsize=12)
    fig.tight_layout(rect=[0, .1, 1, .97])
    fig.savefig(root/"recent_tradeoffs.png", dpi=180, bbox_inches="tight")
    fig.savefig(root/"recent_tradeoffs.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
