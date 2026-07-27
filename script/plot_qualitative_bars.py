import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_row(csv_path, row_index=None, prefer_rescued=False):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    if prefer_rescued:
        for row in rows:
            if int(float(row.get("clip_correct", 0))) == 0 and int(float(row.get("model_correct", 0))) == 1:
                return row

    if row_index is None:
        row_index = 0
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index={row_index} out of range for {len(rows)} rows")
    return rows[row_index]


def maybe_to_percent(values, force_percent=False):
    values = [float(v) for v in values]
    if force_percent or max(abs(v) for v in values) <= 1.5:
        return [v * 100.0 for v in values]
    return values


def plot_bars(clip_true, clip_false, model_true, model_false, output, title=None):
    values = np.array([
        [clip_false, clip_true],
        [model_false, model_true],
    ], dtype=float)

    groups = ["CLIP", "Ours"]
    labels = ["False Caption", "True Caption"]
    colors = ["#efc47f", "#91aecd"]

    x = np.arange(len(groups))
    width = 0.34

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.1,
    })

    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=300)
    bars_false = ax.bar(x - width / 2, values[:, 0], width, label=labels[0], color=colors[0], edgecolor="none")
    bars_true = ax.bar(x + width / 2, values[:, 1], width, label=labels[1], color=colors[1], edgecolor="none")

    for bars in (bars_false, bars_true):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.8,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    ymin = max(0.0, float(values.min()) - 10.0)
    ymax = float(values.max()) + 10.0
    if ymax - ymin < 20:
        ymax = ymin + 20
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontweight="bold")
    ax.set_ylabel("Cosine Similarity (%)", fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold", pad=8)

    ax.grid(axis="y", linestyle=":", linewidth=0.9, color="#bdbdbd")
    ax.set_axisbelow(True)
    ax.legend(
        handles=[bars_true, bars_false],
        labels=["True Caption", "False Caption"],
        loc="upper left",
        frameon=True,
        edgecolor="#555555",
        fontsize=9,
    )

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.1)

    fig.tight_layout(pad=0.6)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved figure to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot CLIP vs structural scorer true/false caption bars.")
    parser.add_argument("--csv", default="", help="CSV exported by export_qualitative_scores_v2.py.")
    parser.add_argument("--row_index", default=0, type=int, help="Row index to plot from CSV.")
    parser.add_argument("--prefer_rescued", action="store_true", help="Use first CLIP-wrong/model-correct row if available.")
    parser.add_argument("--output", default="outputs/qualitative_bar.png")
    parser.add_argument("--title", default="")
    parser.add_argument("--force_percent", action="store_true", help="Multiply scores by 100 even if values look large.")

    parser.add_argument("--clip_true", type=float, default=None)
    parser.add_argument("--clip_false", type=float, default=None)
    parser.add_argument("--model_true", type=float, default=None)
    parser.add_argument("--model_false", type=float, default=None)
    args = parser.parse_args()

    manual_values = [args.clip_true, args.clip_false, args.model_true, args.model_false]
    if all(v is not None for v in manual_values):
        clip_true, clip_false, model_true, model_false = maybe_to_percent(manual_values, args.force_percent)
    else:
        if not args.csv:
            raise ValueError("Provide --csv or all four manual scores.")
        row = load_row(args.csv, row_index=args.row_index, prefer_rescued=args.prefer_rescued)
        clip_true, clip_false, model_true, model_false = maybe_to_percent(
            [row["clip_true"], row["clip_false"], row["model_true"], row["model_false"]],
            args.force_percent,
        )

    plot_bars(
        clip_true=clip_true,
        clip_false=clip_false,
        model_true=model_true,
        model_false=model_false,
        output=args.output,
        title=args.title or None,
    )


if __name__ == "__main__":
    main()


'''
MPLCONFIGDIR=/tmp/matplotlib python script/plot_qualitative_bars.py \
--clip_true 49.93 \
--clip_false 50.07 \
--model_true 61.51 \
--model_false 38.49 \
--output outputs/qualitative_bar_example.png
'''