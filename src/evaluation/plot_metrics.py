"""Figures for the report, made only from the CSV files written by evaluation.py.

    python src/evaluation/plot_metrics.py

Reads   results/metrics/{lights_finetuned_val,signs_finetuned_test}/*.csv
Writes  results/metrics/figures/*.png (and .pdf for LaTeX)

1. Confusion matrix heatmaps (class-agnostic matching, conf 0.25, IoU 0.5).
   Cell text = count. Cell colour = share of its row, so rare classes are as readable as
   frequent ones.
2. Recall by box size. Lights are binned by box WIDTH, signs by box HEIGHT at model input
   (team decision, Option A). The two panels use different definitions and must not be compared.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

METRICS = Path("results/metrics")
OUT = METRICS / "figures"

# colours: one blue ramp for magnitude, two categorical slots for the two series
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = {"class_agnostic": "#2a78d6", "class_aware": "#eb6834"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    # no creation date in the PDF, so re-running gives byte-identical files (no false git changes)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


# ----------------------------------------------------------------------------- 1. confusion
def plot_confusion(folder, title, name):
    cm = pd.read_csv(METRICS / folder / "confusion_matrix.csv", index_col=0)
    share = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)        # each row sums to 1
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)

    n_r, n_c = cm.shape
    fig, ax = plt.subplots(figsize=(0.9 * n_c + 1.6, 0.6 * n_r + 1.2))
    im = ax.imshow(share.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(n_r):
        for j in range(n_c):
            v = cm.iat[i, j]
            if v == 0:
                continue
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if share.iat[i, j] > 0.55 else INK)

    ax.set_xticks(range(n_c), cm.columns, rotation=30, ha="right")
    ax.set_yticks(range(n_r), cm.index)
    ax.set_xlabel("Predicted class   (last column: ground truth not detected)")
    ax.set_ylabel("Ground truth   (last row: false positives)")
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    # thin separators around the FP row and the missed column
    ax.axhline(n_r - 1.5, color="white", lw=2)
    ax.axvline(n_c - 1.5, color="white", lw=2)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Share of row", color=INK_2)
    cb.outline.set_visible(False)
    save(fig, name)


# ----------------------------------------------------------------------------- 2. size
def plot_recall_by_size(panels, name):
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.0), sharey=True)
    width = 0.38
    for ax, (folder, title) in zip(axes, panels):
        for k, (mode, label) in enumerate([("class_agnostic", "Class-agnostic"),
                                           ("class_aware", "Class-aware")]):
            d = pd.read_csv(METRICS / folder / f"recall_by_size_{mode}.csv")
            x = [i + (k - 0.5) * width for i in range(len(d))]
            ax.bar(x, d["recall"], width=width - 0.03, color=SERIES[mode], label=label)
            for xi, r in zip(x, d["recall"]):
                ax.text(xi, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=7, color=INK_2)
        ax.set_xticks(range(len(d)), [f"{b}\n(n={t})" for b, t in zip(d["size_bin"], d["total"])])
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Recall")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0.06, 0.98))
    fig.text(0.01, -0.09,"conf ≥ 0.25, IoU ≥ 0.5. Size definitions differ between panels; "
             "do not compare lights with signs.", fontsize=7, color=MUTED)
    save(fig, name)


if __name__ == "__main__":
    plot_confusion("lights_finetuned_val", "Traffic lights, val (865 images)", "confusion_lights_val")
    plot_confusion("signs_finetuned_test", "Traffic signs, test (218 images)", "confusion_signs_test")
    plot_confusion("signs_finetuned_val", "Traffic signs, val (217 images)", "confusion_signs_val")
    plot_recall_by_size(
        [("lights_finetuned_val", "Traffic lights (val), by box width"),
         ("signs_finetuned_test", "Traffic signs (test), by box height at model input")],
        "recall_by_size")
