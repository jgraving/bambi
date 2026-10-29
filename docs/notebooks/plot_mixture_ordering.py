"""Plot compact mixture-validation summaries from saved JSON without fitting.

Run ``python docs/notebooks/plot_mixture_ordering.py`` to refresh the SVG in
docs/validation. Use --data-dir and --output-dir for a different run, and
--preview-dir for optional PNG previews. Requires Matplotlib, not PyMC or Bambi.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


TEAL = "#007F73"
ORANGE = "#D55E00"
GRAY = "#667085"


def paired_plot(ax, components, first, second, labels, formatter):
    """Show paired results without connecting different component counts."""
    positions = list(range(len(components)))
    ax.hlines(positions, first, second, color="#D0D5DD", linewidth=2, zorder=1)
    ax.scatter(
        first, positions, color=TEAL, marker="o", s=46, label=labels[0], zorder=3
    )
    ax.scatter(
        second, positions, color=ORANGE, marker="s", s=42, label=labels[1], zorder=3
    )
    for values, color, offset in ((first, TEAL, -15), (second, ORANGE, 10)):
        for value, position in zip(values, positions):
            ax.annotate(
                formatter(value),
                (value, position),
                xytext=(0, offset),
                textcoords="offset points",
                color=color,
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_yticks(positions, [f"K = {n}" for n in components])
    ax.set_ylim(len(components) - 0.55, -0.55)
    ax.tick_params(axis="both", length=0, pad=7, labelsize=9, colors=GRAY)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#EAECF0", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.04, 1.06),
        ncols=2,
        frameon=False,
        fontsize=8.3,
        columnspacing=0.8,
        handletextpad=0.35,
    )


def make_figure(validation, comparison):
    """Keep identity and sampling evidence in separate, labelled panels."""
    identity = {(row["components"], row["scenario"]): row for row in validation}
    components = sorted({row["components"] for row in validation})
    away = [
        100
        * identity[n, "away_from_branch_cut"][
            "fraction_draws_with_rank_identity_change"
        ]
        for n in components
    ]
    near = [
        100 * identity[n, "near_branch_cut"]["fraction_draws_with_rank_identity_change"]
        for n in components
    ]
    near_comparison = [
        row for row in comparison if row["scenario"] == "near_branch_cut"
    ]
    sampling = {(row["components"], row["method"]): row for row in near_comparison}
    sampling_components = sorted({row["components"] for row in near_comparison})
    argsort = [sampling[n, "argsort"]["max_rhat_sorted"] for n in sampling_components]
    ordered = [sampling[n, "ordered"]["max_rhat_sorted"] for n in sampling_components]

    fig = plt.figure(figsize=(8.4, 3.55), facecolor="white")
    left = fig.add_axes((0.095, 0.265, 0.355, 0.46))
    right = fig.add_axes((0.585, 0.265, 0.365, 0.46))
    fig.text(
        0.03,
        0.94,
        "Circular mixtures: alignment and sampling are separate checks",
        fontsize=13,
        weight="bold",
        color="#101828",
    )
    fig.text(
        0.095, 0.865, "A  Physical identity after sorting", fontsize=11, weight="bold"
    )
    fig.text(0.585, 0.865, "B  Sampling near ±π", fontsize=11, weight="bold")

    paired_plot(
        left,
        components,
        away,
        near,
        ("Away from cut", "Near cut"),
        lambda x: f"{x:.2f}%",
    )
    left.set_xlim(-2, max(away + near) + 4)
    left.xaxis.set_major_locator(MultipleLocator(5))
    left.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    left.set_xlabel("Draws with a rank/identity mismatch", fontsize=9, labelpad=9)

    paired_plot(
        right,
        sampling_components,
        argsort,
        ordered,
        ("Circular + argsort", "Bounded ordered"),
        lambda x: f"{x:.3f}",
    )
    right.set_xlim(min(1, *argsort, *ordered) - 0.07, max(argsort + ordered) + 0.12)
    right.xaxis.set_major_locator(MultipleLocator(0.2))
    right.set_xlabel("Maximum sorted R-hat", fontsize=9, labelpad=9)

    density_error = max(
        row["max_log_density_difference_after_sorting"] for row in validation
    )
    divergences = sum(row["divergences"] for row in near_comparison)
    fig.text(
        0.03,
        0.095,
        f"A: max |Δ log density| = {density_error:.1e} across {len(validation)} fits. "
        f"B: {divergences} divergences across {len(near_comparison)} near-boundary fits.",
        fontsize=8.5,
        color=GRAY,
    )
    fig.text(
        0.03,
        0.035,
        "One simulated dataset per K. These cases do not establish a general trend with K "
        "or stable physical labels.",
        fontsize=8.5,
        color=GRAY,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "validation",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    validation = json.loads(
        (args.data_dir / "mixture_ordering_results.json").read_text()
    )
    comparison = json.loads(
        (args.data_dir / "mixture_ordering_comparison.json").read_text()
    )
    output = args.output_dir or args.data_dir
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "mixture-ordering",
        }
    ):
        fig = make_figure(validation, comparison)
        svg_path = output / "mixture_ordering_overview.svg"
        fig.savefig(
            svg_path,
            metadata={"Date": None, "Creator": "Bambi mixture validation plots"},
        )
        # Preserve newline separators within path attributes while trimming spaces.
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
            + "\n"
        )
        if args.preview_dir is not None:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / "mixture_ordering_overview.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
