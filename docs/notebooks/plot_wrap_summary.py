"""Plot the wrap experiment's main findings from saved results, without sampling.

Run from the repository root::

    python docs/notebooks/plot_wrap_summary.py

Only JSON, NumPy, and Matplotlib are needed. Use ``--preview path.png`` to also
render a PNG for inspection. The default input records the original modulo
implementation; this figure does not validate subsequent roundoff refinements.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

TEAL = "#007F73"
ORANGE = "#D55E00"
GRAY = "#667085"
SVG_STYLE = {"svg.fonttype": "none", "svg.hashsalt": "bambi-wrap-summary"}


def plot_summary(results):
    """Return a two-panel figure using the recorded diagnostics and step counts."""
    cases = (
        ("intercept_aliases", 0.0, "Same direction\nacross periods"),
        ("slope_aliases", 1.0, "Aliased slopes\nobserved x = 1"),
        ("slope_aliases", 0.5, "Aliased slopes\nnew x = 0.5"),
    )
    fits = results["fits"]
    rhats = []
    for scenario, probe, _ in cases:
        selected = sorted(
            (
                fit
                for fit in fits
                if fit["scenario"] == scenario and fit["link"] == "identity"
            ),
            key=lambda fit: fit["seed"],
        )
        values = np.array(
            [
                [fit[metric]["rhat"][fit["probe_x"].index(probe)] for fit in selected]
                for metric in ("reported_mu", "posthoc_wrapped_mu")
            ]
        )
        if not selected or not np.isfinite(values).all():
            raise ValueError(
                f"Missing or nonfinite R-hat values for {scenario} at x={probe}."
            )
        rhats.append(values)

    paired = {}
    for fit in fits:
        key = (fit["scenario"], fit["seed"])
        if fit["link"] in paired.setdefault(key, {}):
            raise ValueError(f"Duplicate fit for {key}, link={fit['link']}.")
        paired[key][fit["link"]] = fit["total_nuts_steps"]
    if any(set(pair) != {"identity", "wrap"} for pair in paired.values()):
        raise ValueError(
            "Each scenario and seed must have one identity and one wrap fit."
        )
    counts = np.array([[pair["identity"], pair["wrap"]] for pair in paired.values()])
    equal_pairs = int(np.count_nonzero(counts[:, 0] == counts[:, 1]))
    divergences = sum(fit["divergences"] for fit in fits)
    seeds = len({fit["seed"] for fit in fits})

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": "#202B3B",
        "axes.labelcolor": GRAY,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "axes.edgecolor": "#CBD0D8",
        "axes.spines.top": False,
        "axes.spines.right": False,
        **SVG_STYLE,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(8.6, 4.0))
        left = fig.add_axes((0.205, 0.26, 0.365, 0.50))
        right = fig.add_axes((0.72, 0.26, 0.25, 0.50))
        fig.text(
            0.025,
            0.94,
            "Directional diagnostics change; sampling work stays the same",
            fontsize=13,
            weight="bold",
        )
        fig.text(
            0.025,
            0.875,
            f"Original modulo experiment  ·  {len(fits)} fits  ·  {seeds} data seeds",
            fontsize=10,
            color=GRAY,
        )
        left.set_title(
            "A  Same draws, two representations", loc="left", fontsize=10, pad=14
        )
        for row, values in enumerate(rhats):
            medians = np.median(values, axis=1)
            left.plot(medians, [row, row], color="#CBD0D8", lw=1.5, zorder=1)
            for index, (color, marker, label) in enumerate(
                ((ORANGE, "s", "Raw predictor"), (TEAL, "o", "Wrapped direction"))
            ):
                median = medians[index]
                errors = [
                    [median - values[index].min()],
                    [values[index].max() - median],
                ]
                left.errorbar(
                    median,
                    row,
                    xerr=errors,
                    fmt=marker,
                    ms=6.5,
                    color=color,
                    capsize=3,
                    label=label if row == 0 else None,
                    zorder=3,
                )
                left.annotate(
                    f"{median:.3f}",
                    (median, row),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=9,
                )
        left.axvline(1, color="#CBD0D8", ls=":", lw=1, zorder=0)
        left.set(
            yticks=range(len(cases)),
            yticklabels=[case[2] for case in cases],
            ylim=(len(cases) - 0.55, -0.55),
            xlim=(0.86, 3.02),
            xticks=[1, 1.5, 2, 2.5, 3],
            xlabel="Rank R-hat",
        )
        left.tick_params(axis="y", length=0, pad=10)
        left.spines["left"].set_visible(False)
        left.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.30),
            ncol=2,
            frameon=False,
            fontsize=9,
            handlelength=1,
            columnspacing=1.4,
        )

        right.set_title("B  Matched sampling work", loc="left", fontsize=10, pad=14)
        thousands = counts / 1000
        lower, upper = float(thousands.min()) - 0.7, float(thousands.max()) + 0.7
        right.plot([lower, upper], [lower, upper], color="#CBD0D8", ls="--", lw=1)
        right.scatter(thousands[:, 0], thousands[:, 1], s=26, color=TEAL, zorder=3)
        right.set(
            xlim=(lower, upper),
            ylim=(lower, upper),
            xlabel="Identity: NUTS steps (×1,000)",
            ylabel="Wrap: NUTS steps (×1,000)",
        )
        right.text(
            0.04,
            0.95,
            f"{equal_pairs}/{len(paired)} pairs\nidentical counts",
            transform=right.transAxes,
            va="top",
            fontsize=10,
        )
        right.text(
            0.5,
            -0.38,
            f"{divergences} divergences across {len(fits)} fits",
            transform=right.transAxes,
            ha="center",
            fontsize=9,
            color=GRAY,
        )
    return fig


def main():
    default_dir = Path(__file__).resolve().parents[1] / "validation" / "wrap"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_dir / "results.json")
    parser.add_argument("--output", type=Path, default=default_dir / "wrap_summary.svg")
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    figure = plot_summary(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(SVG_STYLE):
        figure.savefig(
            args.output,
            format="svg",
            metadata={"Date": None, "Creator": "Bambi wrap summary"},
        )
    # Preserve path-attribute line breaks while removing Matplotlib's trailing spaces.
    svg = args.output.read_text(encoding="utf-8")
    args.output.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8"
    )
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.preview, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
