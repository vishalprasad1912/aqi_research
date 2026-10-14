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
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_mse_scaled"], label="Train loss", color="tab:blue")
    ax.plot(history["epoch"], history["val_mse_scaled"], label="Validation loss", color="tab:orange")
    if best_epoch is not None:
        ax.axvline(best_epoch, ls="--", color="gray", label=f"Best model (epoch {best_epoch})")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=4)
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_prediction_windows(past, observed, predicted, path, n_samples=6, seed=0,
                            title="Test predictions with input history"):
    n, T = past.shape
    H = observed.shape[1]
    idx = np.sort(np.random.default_rng(seed).choice(n, size=min(n_samples, n), replace=False))
    cols = 3
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    t_past, t_fut = np.arange(-T, 0), np.arange(1, H + 1)        # hours relative to forecast origin
    for ax, i in zip(axes.ravel(), idx):
        ax.plot(t_past, past[i], "o-", color="gray", label="Previous (input)")
        # join the last known value to the forecast horizon so the line reads as one series
        ax.plot(np.r_[t_past[-1], t_fut], np.r_[past[i, -1], observed[i]], "o-",
                color="tab:green", label="Real")
        ax.plot(np.r_[t_past[-1], t_fut], np.r_[past[i, -1], predicted[i]], "s--",
                color="tab:red", label="Predicted")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set(title=f"Test sample {i}", xlabel="Hours relative to forecast time", ylabel="AQI")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3)
    fig.suptitle(title, y=1.0)
    _save(fig, path)


def plot_test_series(observed, predicted, path, times=None, horizon_step=0,
                     title="Test set: real vs predicted", plot_steps=20):
    real_colors = ["#1f77b4", "#2ca02c"]
    pred_colors = ["#6baed6", "#74c476"]

    x = pd.to_datetime(times[::plot_steps]) if times is not None else np.arange(len(observed[::plot_steps, horizon_step]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(x, observed[::plot_steps, horizon_step, 0], lw=1.2, color=real_colors[0],
             label="PM2.5 Real")
    ax1.plot(x, predicted[::plot_steps, horizon_step, 0], lw=1.2, ls='--', color='tab:red',
             label=f"PM2.5 Pred (+{horizon_step + 1} h)")

    ax2.plot(x, observed[::plot_steps, horizon_step, 1], color=real_colors[1],
             label="PM10 Real")
    ax2.plot(x, predicted[::plot_steps, horizon_step, 1], '--', color='tab:red',
             label=f"PM10 Pred (+{horizon_step + 1} h)")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_ylabel("Concentration in (µg/m³)", fontsize=12, fontweight="bold")
    if times is not None:
        rng = f"{pd.Timestamp(x[0]):%Y-%m-%d %H:%M} to {pd.Timestamp(x[-1]):%Y-%m-%d %H:%M}"
        ax1.set_title(f"{title}\n{rng}", fontsize=20, fontweight="bold", pad=4)
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        # ax2.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        # ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        fig.autofmt_xdate()
        ax2.set_xlabel("Timestamp", fontsize=16, fontweight="bold")
        ax2.tick_params(axis='x', labelrotation=90)
    else:
        ax1.set_title(title, fontsize=20, fontweight="bold", pad=4)
        ax2.set_xlabel("Test hour", fontsize=16, fontweight="bold")
    _save(fig, path)