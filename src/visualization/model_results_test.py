from pathlib import Path

import matplotlib
import pandas as pd
import matplotlib.dates as mdates

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_loss_curves(history, path, best_epoch=None, title="Training and validation loss"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["train_mse_scaled"], label="Train loss", color="tab:blue")
    ax.plot(history["epoch"], history["val_mse_scaled"], label="Validation loss", color="tab:orange")
    if best_epoch is not None:
        ax.axvline(best_epoch, ls="--", color="gray", label=f"Best model (epoch {best_epoch})")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold", labelpad=4)
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold", labelpad=4)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)

def plot_test_series(observed, predicted, path, times=None, horizon_step=0,
                     title="Test set: real vs predicted", plot_steps=8):
    real_colors = ["#1f77b4", "#2ca02c"]
    pred_colors = ["#6baed6", "#74c476"]

    x = pd.to_datetime(times[::plot_steps]) if times is not None else np.arange(len(observed[::plot_steps, horizon_step]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(x, observed[::plot_steps, horizon_step, 0], lw=1.2, color=real_colors[0], label="PM2.5 Real")
    ax1.plot(x, predicted[::plot_steps, horizon_step, 0], lw=1.2, ls='--', color="tab:red", label=f"PM2.5 Pred")

    ax2.plot(x, observed[::plot_steps, horizon_step, 1], color=real_colors[1], label="PM10 Real")
    ax2.plot(x, predicted[::plot_steps, horizon_step, 1], '--', color="tab:red", label=f"PM10 Pred")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_ylabel("Concentration in (µg/m³)", fontsize=12, fontweight="bold", labelpad=4)
    if times is not None:
        rng = f"{pd.Timestamp(x[0]):%Y-%m-%d %H:%M} to {pd.Timestamp(x[-1]):%Y-%m-%d %H:%M}"
        ax1.set_title(f"{title}\n{rng}", fontsize=16, fontweight="bold", pad=8)
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        # ax2.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        # ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        fig.autofmt_xdate()
        ax2.set_xlabel("Timestamp", fontsize=12, fontweight="bold", labelpad=4)
        ax2.tick_params(axis='x', labelrotation=90)
    else:
        ax1.set_title(title, fontsize=16, fontweight="bold", pad=8)
        ax2.set_xlabel("Test hour", fontsize=12, fontweight="bold", labelpad=4)
    _save(fig, path)