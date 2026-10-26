"""
generate_plots.py

Creates report-quality figures from the results of augmentation_svm.py
(augmentation_results.npy). Saves 300-DPI PNGs into a 'plots/' folder
next to this script.

Figures (each made for Arousal and Valence):
    1. Grouped bar chart  - accuracy per subject, best ratio per augmentation
    2. Heatmap            - per-subject accuracy, green (high) to red (low)
    3. Improvement chart  - accuracy change vs baseline (+ a separate SMOTE chart)
    4. Ratio comparison   - baseline vs 2x vs 4x for each augmentation

Usage:
    python generate_plots.py                      # uses DATA_FOLDER below
    python generate_plots.py "C:\\path\\to\\folder"  # override the folder
"""

import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # save to files only; no window needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FOLDER = r"C:\Users\JAMSHID KHALIQ\Desktop\Project-EEGResearch\JAMSHID\practise DEAP dataset\data_preprocessed_python"
if len(sys.argv) > 1:
    DATA_FOLDER = sys.argv[1]

RESULTS_FILE = "augmentation_results.npy"
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
DPI = 300

TASKS = ["arousal", "valence"]
SUBJECTS = [f"s{i:02d}" for i in range(1, 11)]

# Display name -> technique prefix used in augmentation_results.npy
RATIO_AUGS = OrderedDict([
    ("Noise", "gaussian_noise"),
    ("Dropout", "channel_dropout"),
    ("Band Masking", "band_masking"),
    ("Mixup", "mixup"),
])
COLUMNS = ["Baseline", "Noise", "Dropout", "Band Masking", "SMOTE", "Mixup"]

COLORS = {
    "Baseline": "#7f7f7f",
    "Noise": "#1f77b4",
    "Dropout": "#ff7f0e",
    "Band Masking": "#9467bd",
    "SMOTE": "#d62728",
    "Mixup": "#2ca02c",
}
HATCHES = {"Noise": "", "Dropout": "//", "Band Masking": "..", "Mixup": "xx"}
GOOD, BAD = "#2e9e4f", "#d64545"

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_results():
    path = os.path.join(DATA_FOLDER, RESULTS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}\n"
            "Run augmentation_svm.py first, or pass the folder as an argument."
        )
    return np.load(path, allow_pickle=True).item()["per_subject"]


def accuracy(per_subject, subject, task, technique):
    """Accuracy in percent."""
    return 100.0 * float(per_subject[subject][task][technique]["accuracy"])


def accuracy_array(per_subject, task, technique):
    return np.array([accuracy(per_subject, s, task, technique) for s in SUBJECTS])


def best_ratio(per_subject, task, prefix):
    """Ratio ('2x' or '4x') with the higher mean accuracy; ties go to 2x."""
    mean_2x = accuracy_array(per_subject, task, f"{prefix}_2x").mean()
    mean_4x = accuracy_array(per_subject, task, f"{prefix}_4x").mean()
    return "4x" if mean_4x > mean_2x else "2x"


def build_task_table(per_subject, task):
    """
    Returns (table, ratio_labels):
        table[column] -> array of 10 accuracies (%), using the best ratio
        ratio_labels[column] -> e.g. 'Noise (4x)'
    """
    table, labels = OrderedDict(), OrderedDict()
    table["Baseline"] = accuracy_array(per_subject, task, "baseline")
    labels["Baseline"] = "Baseline"

    for display, prefix in RATIO_AUGS.items():
        ratio = best_ratio(per_subject, task, prefix)
        table[display] = accuracy_array(per_subject, task, f"{prefix}_{ratio}")
        labels[display] = f"{display} ({ratio.replace('x', '×')})"

    table["SMOTE"] = accuracy_array(per_subject, task, "smote")
    labels["SMOTE"] = "SMOTE"

    # keep the requested column order
    table = OrderedDict((c, table[c]) for c in COLUMNS)
    labels = OrderedDict((c, labels[c]) for c in COLUMNS)
    return table, labels


def save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 1: grouped bar chart
# ---------------------------------------------------------------------------

def plot_grouped_bars(table, labels, task):
    fig, ax = plt.subplots(figsize=(15, 6.5))
    n_groups = len(COLUMNS)
    width = 0.8 / n_groups
    x = np.arange(len(SUBJECTS))

    for i, col in enumerate(COLUMNS):
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, table[col], width, label=labels[col],
               color=COLORS[col], edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Comparison of Augmentation Techniques for {task.capitalize()}",
                 fontsize=16, fontweight="bold", pad=14)
    ax.legend(title="Technique (best ratio)", ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, f"plot1_grouped_bar_{task}.png")


# ---------------------------------------------------------------------------
# Plot 2: heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(table, task):
    data = np.column_stack([table[c] for c in COLUMNS])  # subjects x techniques
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        data, annot=True, fmt=".1f", cmap="RdYlGn",
        vmin=np.floor(data.min()), vmax=np.ceil(data.max()),
        xticklabels=COLUMNS, yticklabels=SUBJECTS,
        linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Accuracy (%)"}, ax=ax,
    )
    ax.set_title(f"Per-Subject Performance Heatmap - {task.capitalize()}",
                 fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Technique")
    ax.set_ylabel("Subject")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    save(fig, f"plot2_heatmap_{task}.png")


# ---------------------------------------------------------------------------
# Plot 3: improvement over baseline
# ---------------------------------------------------------------------------

def plot_improvement(table, labels, task):
    techniques = list(RATIO_AUGS.keys())
    n = len(techniques)
    width = 0.8 / n
    x = np.arange(len(SUBJECTS))

    fig, ax = plt.subplots(figsize=(15, 6.5))
    for i, tech in enumerate(techniques):
        diff = table[tech] - table["Baseline"]
        offset = (i - (n - 1) / 2) * width
        colors = [GOOD if d >= 0 else BAD for d in diff]
        ax.bar(x + offset, diff, width, color=colors, hatch=HATCHES[tech],
               edgecolor="white", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Improvement over Baseline (%)")
    ax.set_title(f"Improvement Over Baseline by Augmentation Type - {task.capitalize()}",
                 fontsize=16, fontweight="bold", pad=14)

    handles = [Patch(facecolor=GOOD, label="Helped (+)"),
               Patch(facecolor=BAD, label="Hurt (-)")]
    handles += [Patch(facecolor="#bbbbbb", edgecolor="white", hatch=HATCHES[t],
                      label=labels[t]) for t in techniques]
    ax.legend(handles=handles, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False,
              title="Colour = effect, pattern = technique (best ratio)")
    ax.grid(axis="x", visible=False)
    save(fig, f"plot3_improvement_{task}.png")


def plot_smote_improvement(table, task):
    diff = table["SMOTE"] - table["Baseline"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(SUBJECTS, diff, color=[GOOD if d >= 0 else BAD for d in diff],
           edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Improvement over Baseline (%)")
    ax.set_title(f"SMOTE Improvement Over Baseline - {task.capitalize()}",
                 fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="x", visible=False)
    save(fig, f"plot3b_smote_improvement_{task}.png")


# ---------------------------------------------------------------------------
# Plot 4: ratio comparison
# ---------------------------------------------------------------------------

def plot_ratio_comparison(per_subject, task):
    baseline = accuracy_array(per_subject, task, "baseline")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)

    for ax, (display, prefix) in zip(axes.ravel(), RATIO_AUGS.items()):
        ax.plot(SUBJECTS, baseline, color="#7f7f7f", linestyle="--",
                marker="o", linewidth=2, label="Baseline")
        ax.plot(SUBJECTS, accuracy_array(per_subject, task, f"{prefix}_2x"),
                color="#1f77b4", marker="s", linewidth=2, label="2× Ratio")
        ax.plot(SUBJECTS, accuracy_array(per_subject, task, f"{prefix}_4x"),
                color="#d62728", marker="^", linewidth=2, label="4× Ratio")
        ax.set_title(display, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Accuracy (%)")

    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Impact of Augmentation Ratio - {task.capitalize()}",
                 fontsize=17, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    save(fig, f"plot4_ratio_comparison_{task}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    per_subject = load_results()
    print(f"Saving plots to {PLOTS_DIR}")

    for task in TASKS:
        print(f"\n{task.capitalize()}:")
        table, labels = build_task_table(per_subject, task)
        print("  best ratios: " + ", ".join(
            labels[c] for c in COLUMNS if c in RATIO_AUGS))
        plot_grouped_bars(table, labels, task)
        plot_heatmap(table, task)
        plot_improvement(table, labels, task)
        plot_smote_improvement(table, task)
        plot_ratio_comparison(per_subject, task)

    print("\nDone.")


if __name__ == "__main__":
    main()