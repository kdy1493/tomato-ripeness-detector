"""
Plot training metrics from results.csv.

Usage:
  python3 scripts/plot_results.py --csv runs/yolo26_custom_tomato/results.csv
  python3 scripts/plot_results.py --csv runs/yolo26_custom_tomato/results.csv --out runs/yolo26_custom_tomato/metrics.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(csv_path: str, out_path: str | None = None) -> None:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    epochs = df["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("YOLO26 Training Metrics", fontsize=16, fontweight="bold")

    # Train losses
    ax = axes[0, 0]
    ax.plot(epochs, df["train/box_loss"], label="box_loss", linewidth=1.5)
    ax.plot(epochs, df["train/cls_loss"], label="cls_loss", linewidth=1.5)
    ax.plot(epochs, df["train/dfl_loss"], label="dfl_loss", linewidth=1.5)
    ax.set_title("Train Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Val losses
    ax = axes[0, 1]
    ax.plot(epochs, df["val/box_loss"], label="box_loss", linewidth=1.5)
    ax.plot(epochs, df["val/cls_loss"], label="cls_loss", linewidth=1.5)
    ax.plot(epochs, df["val/dfl_loss"], label="dfl_loss", linewidth=1.5)
    ax.set_title("Val Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Train vs Val box_loss (overfitting check)
    ax = axes[0, 2]
    ax.plot(epochs, df["train/box_loss"], label="train", linewidth=1.5)
    ax.plot(epochs, df["val/box_loss"], label="val", linewidth=1.5)
    ax.fill_between(epochs, df["train/box_loss"], df["val/box_loss"], alpha=0.15, color="red")
    ax.set_title("Box Loss: Train vs Val (overfit gap)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision & Recall
    ax = axes[1, 0]
    ax.plot(epochs, df["metrics/precision(B)"], label="Precision", linewidth=1.5)
    ax.plot(epochs, df["metrics/recall(B)"], label="Recall", linewidth=1.5)
    ax.set_title("Precision & Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mAP
    ax = axes[1, 1]
    ax.plot(epochs, df["metrics/mAP50(B)"], label="mAP50", linewidth=1.5)
    ax.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP50-95", linewidth=1.5)
    ax.set_title("mAP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    best_epoch = df["metrics/mAP50-95(B)"].idxmax()
    best_map = df.loc[best_epoch, "metrics/mAP50-95(B)"]
    ax.axvline(x=df.loc[best_epoch, "epoch"], color="red", linestyle="--", alpha=0.5)
    ax.annotate(f"best: {best_map:.4f} (ep {int(df.loc[best_epoch, 'epoch'])})",
                xy=(df.loc[best_epoch, "epoch"], best_map),
                xytext=(10, -20), textcoords="offset points",
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.7))

    # Learning rate
    ax = axes[1, 2]
    ax.plot(epochs, df["lr/pg0"], label="lr/pg0", linewidth=1.5)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path is None:
        out_path = str(Path(csv_path).parent / "metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    plot_results(args.csv, args.out)


if __name__ == "__main__":
    main()
