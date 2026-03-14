import argparse

import matplotlib.pyplot as plt
import numpy as np

# parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-title",
    action="store_true",
    help="If set, the plot will be generated without a title.",
)
args = parser.parse_args()
no_title = args.no_title
# -----------------------
# Data
# -----------------------
models = [
    "Qwen2.5-14B",
    "Qwen2.5-7B",
    "Qwen2.5-3B",
    "Qwen2.5-1.5B",
    "Qwen2.5-0.5B",
]

correctness = [4.04, 4.01, 3.96, 1.29, 0.71]
completeness = [3.94, 3.66, 3.56, 1.04, 0.79]
clarity = [3.96, 3.91, 3.86, 2.79, 2.54]

# -----------------------
# Styling
# -----------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

# Colorblind-safe palette
colors = ["#4C72B0", "#55A868", "#C44E52"]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 4.8))

# Bars
ax.bar(x - width, correctness, width, label="Correctness", color=colors[0])
ax.bar(x, completeness, width, label="Completeness", color=colors[1])
ax.bar(x + width, clarity, width, label="Clarity", color=colors[2])

if highlight_baseline := False:
    # -----------------------
    # Highlight 14B baseline
    # -----------------------
    baseline_values = {
        "Correctness": correctness[0],
        "Completeness": completeness[0],
        "Clarity": clarity[0],
    }

    for value, color, label in zip(
        baseline_values.values(),
        colors,
        baseline_values.keys(),
    ):
        ax.axhline(
            y=value,
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.9,
        )
        # ax.text(
        #     x=len(models) - 0.35,
        #     y=value + 0.05,
        #     s=f"{label}: {value:.2f}",
        #     color=color,
        #     fontsize=9,
        #     ha="right",
        #     va="bottom",
        # )

    # -----------------------
    # Annotate 14B bars
    # -----------------------
    baseline_x = x[0]

    baseline_values = [
        correctness[0],
        completeness[0],
        clarity[0],
    ]

    offsets = [-width, 0, width]

    for value, dx in zip(baseline_values, offsets):
        ax.text(
            baseline_x + dx,
            value + 0.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


# Axes & grid
ax.set_ylabel("Score (out of 5)")
ax.set_xlabel("Model")
ax.set_ylim(0, 4.5)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)

ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
ax.legend(frameon=False, loc="upper right")

# Title
if not no_title:
    ax.set_title("Overall Evaluation Across Models")

plt.tight_layout()
plt.savefig("overall_evaluation.pdf", bbox_inches="tight")
plt.savefig("overall_evaluation.png", dpi=300, bbox_inches="tight")
plt.show()
