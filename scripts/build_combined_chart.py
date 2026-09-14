from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/assets/combined/scores_by_income.png"


def main() -> None:
    with (ROOT / "results/combined_final/benchmark_by_income.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    values = {
        (r["model"], r["task"], r["income_quartile"]): 100 * float(r["primary_score"])
        for r in rows
    }
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    groups = [
        ("CLIP", "clip", "classification"),
        ("Qwen", "qwen", "classification"),
        ("InternVL", "internvl", "classification"),
        ("YOLO-World", "yolo_world", "detection"),
        ("BLIP", "blip_baseline", "captioning"),
        ("BLIP + prompt", "blip_prompted", "captioning"),
        ("Qwen", "qwen", "captioning"),
        ("InternVL", "internvl", "captioning"),
    ]
    colors = ["#16697A", "#DB7C26", "#4C956C", "#7D5BA6"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.35), dpi=220)
    for ax, selected, title in (
        (axes[0], groups[:4], "Classification / detection"),
        (axes[1], groups[4:], "Automatic caption recall"),
    ):
        for idx, (label, model, task) in enumerate(selected):
            ax.plot(
                quartiles,
                [values[(model, task, quartile)] for quartile in quartiles],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                color=colors[idx],
                label=label,
            )
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (%)", fontsize=8)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="lower center")
    fig.tight_layout(pad=0.7)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
