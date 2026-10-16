"""Figures derived only from exported metrics, predictions and genuine image arrays."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cowbias.evaluation.metrics import GROUPS, MODELS

COLORS = ("#4477aa", "#228833")
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180}
)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def seed_figures(root: Path):
    metrics = json.loads((root / "metrics.json").read_text())
    with np.load(root / "test_inputs.npz", allow_pickle=False) as f:
        images, ids = f["images"], f["image_ids"].tolist()
    rows = metrics["models"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(4)
    for i, model in enumerate(MODELS):
        vals = [rows[model]["groups"][g]["accuracy"] for g in GROUPS]
        ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, label=model, color=COLORS[i])
    ax.set(
        xticks=x,
        xticklabels=[g.replace("/", "\n") for g in GROUPS],
        ylim=(0, 1.07),
        ylabel="Held-out accuracy",
        title=f"SYNTHETIC — seed {metrics['seed']}; n/group = "
        f"{rows['baseline']['groups'][GROUPS[0]]['total']}",
    )
    ax.legend(loc="lower right")
    save(fig, root / "figures/accuracy_by_group.png")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    vmax = max(max(max(r) for r in rows[m]["confusion_matrix"]) for m in MODELS)
    for ax, m in zip(axes, MODELS, strict=True):
        matrix = np.array(rows[m]["confusion_matrix"])
        ax.imshow(matrix, vmin=0, vmax=vmax, cmap="Blues")
        for y in range(2):
            for x in range(2):
                ax.text(
                    x,
                    y,
                    str(matrix[y, x]),
                    ha="center",
                    va="center",
                    color="white" if matrix[y, x] > vmax * 0.6 else "black",
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["non_cow", "cow"],
            yticklabels=["non_cow", "cow"],
            xlabel="Predicted",
            ylabel="Actual",
            title=m,
        )
    fig.suptitle(f"SYNTHETIC held-out confusion — seed {metrics['seed']}")
    save(fig, root / "figures/confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(MODELS):
        cf = rows[m]["counterfactual"]
        ax.bar(
            np.arange(2) + (i - 0.5) * 0.36,
            [cf["background_flip_rate"], cf["paired_correct_rate"]],
            0.36,
            label=m,
            color=COLORS[i],
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Background flip rate", "Both images correct"],
        ylim=(0, 1.07),
        ylabel="Fraction of true foreground pairs",
        title=f"SYNTHETIC — seed {metrics['seed']}; "
        f"{rows['baseline']['counterfactual']['total_pairs']} pairs/model",
    )
    ax.legend()
    save(fig, root / "figures/background_sensitivity.png")
    import csv

    with (root / "predictions.csv").open() as f:
        pred = {(r["model"], r["image_id"]): r for r in csv.DictReader(f)}
    with (root / "gradcam_examples.csv").open() as f:
        selected = list(csv.DictReader(f))
    heats = {}
    for m in MODELS:
        with np.load(root / f"{m}_heatmaps.npz", allow_pickle=False) as f:
            heats[m] = f["heatmaps"]
    for selection, filename, cams in [
        (
            [r for r in selected if r["selection"] == "fixed_identity"],
            "gradcam_comparison.png",
            True,
        ),
        (
            [r for r in selected if r["selection"] == "fixed_identity"],
            "counterfactual_examples.png",
            False,
        ),
        (
            [r for r in selected if r["selection"] != "fixed_identity"],
            "exploratory_cases.png",
            True,
        ),
    ]:
        if not selection:
            continue
        fig, axes = plt.subplots(
            len(selection), 3, figsize=(10, len(selection) * 2.5), squeeze=False
        )
        for row_axes, example in zip(axes, selection, strict=True):
            image_id = example["image_id"]
            index = ids.index(image_id)
            img = images[index]
            row_axes[0].imshow(img)
            row_axes[0].set_title(
                f"{example['object_label']} / {example['background_label']}\n{image_id}", fontsize=9
            )
            for j, m in enumerate(MODELS, 1):
                ax = row_axes[j]
                ax.imshow(img)
                if cams:
                    ax.imshow(
                        heats[m][index],
                        cmap="inferno",
                        vmin=0,
                        vmax=1,
                        alpha=0.50,
                        interpolation="nearest",
                    )
                p = pred[m, image_id]
                label = "cow" if int(p["predicted"]) else "non_cow"
                ax.set_title(
                    f"{m}: {label}; P(cow)={float(p['prob_cow']):.4f}\n"
                    f"true-target={example['object_label']}",
                    fontsize=9,
                )
            for ax in row_axes:
                ax.axis("off")
        label = (
            "FIXED IDENTITY SAMPLE"
            if selection[0]["selection"] == "fixed_identity"
            else "EXPLORATORY OUTCOME CASES"
        )
        fig.suptitle(
            f"SYNTHETIC — seed {metrics['seed']} — {label}\n"
            + (
                "Grad-CAM: each map peak-normalized [0,1]; overlay alpha 0.50"
                if cams
                else "Each adjacent grass/beach row shares its foreground identity"
            ),
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, root / "figures" / filename)


def cross_seed_figure(root: Path):
    summary = json.loads((root / "summary.json").read_text())
    seeds = summary["seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in zip(
        axes,
        ("overall_accuracy", "worst_group_accuracy"),
        ("Overall accuracy", "Worst-group accuracy"),
        strict=True,
    ):
        for i, m in enumerate(MODELS):
            r = next(r for r in summary["statistics"] if r["model"] == m and r["metric"] == metric)
            x = np.arange(len(seeds)) + (i - 0.5) * 0.16
            ax.plot(x, [r["values"][str(s)] for s in seeds], "o-", color=COLORS[i], label=m)
            ax.errorbar(
                len(seeds) + (i - 0.5) * 0.16,
                r["mean"],
                yerr=r["sample_sd"] or 0,
                fmt="s",
                color=COLORS[i],
                capsize=5,
                clip_on=False,
            )
        ax.set(
            xticks=range(len(seeds) + 1),
            xticklabels=[str(s) for s in seeds] + ["Mean ± SD"],
            ylim=(0, 1.10),
            ylabel=title,
            xlabel="Experiment seed (joint data + initialization)",
        )
        ax.legend(loc="lower right")
    fig.suptitle(
        "SYNTHETIC held-out results — sample SD (ddof=1), three seeds; no significance claim"
    )
    fig.tight_layout()
    save(fig, root / "figures/cross_seed_summary.png")
