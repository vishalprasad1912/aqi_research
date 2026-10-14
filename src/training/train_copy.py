"""Train / validate / test LSTM, GCN-LSTM or GLSTM-Block on one city.

Run from the project root:
    python -m src.training.train --model lstm         --city beijing
    python -m src.training.train --model gcn_lstm     --city shanghai
    python -m src.training.train --model glstm_block  --city shenzhen --horizon 1 --input_steps 3

Expected input file  data/processed/<city>.npz  with arrays:
    values     (T_total, N, F) float  hourly features per station (pollutants + weather, raw units,
                                      missing values already interpolated)
    coords     (N, 2)                 (lat, lon) of each station
    timestamps (T_total,)             datetime-like (str or datetime64)
    target_idx scalar int             index of the target station (e.g. 1006A / 1147A / 1358A)
    aqi_col    scalar int             feature index of AQI

Protocol (paper Sec. 2.3 / 4.1): chronological 70/15/15 split without shuffling, min-max scaling,
Adam, batch 32, dropout 0.5, 100 epochs, best weights kept by validation RMSE, early stopping.
Metrics (RMSE, MAE, IA, R2) are computed in original AQI units on the test set.
"""
import argparse
import csv
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_all
from src.models.baselines.gcn_lstm import GCNLSTM
from src.models.baselines.lstm import LSTMModel
from src.models.model_utils import build_adjacency, count_parameters
from src.models.proposed.glstm_block import GLSTMBlockModel
from src.utils.paths import CONFIG, MODELS_SAVED, PROCESSED, RESULTS, SPLITS
from src.utils.seed import set_seed
from src.visualization.result_plots import (plot_loss_curves, plot_prediction_windows,
                                             plot_test_series)

MODELS = {
    "lstm": (LSTMModel, "baselines"),
    "gcn_lstm": (GCNLSTM, "baselines"),
    "glstm_block": (GLSTMBlockModel, "proposed"),
}


# ----------------------------------------------------------------------------- data
def load_city(city):
    d = np.load(PROCESSED / f"{city}.npz", allow_pickle=True)
    return (d["values"].astype(np.float32), d["coords"], pd.to_datetime(d["timestamps"]),
            int(d["target_idx"]), int(d["aqi_col"]))


def time_codes(ts):
    """(T_total, 3) long: hour 0-23, weekday 0-6, month 0-11 (for the embedding layer)."""
    return np.stack([ts.hour, ts.dayofweek, ts.month - 1], axis=1).astype(np.int64)


def make_windows(values, tcodes, aqi_col, target_idx, T, H):
    """Sample i: input rows [i, i+T), target = target-station AQI at rows [i+T, i+T+H)."""
    n = len(values) - T - H + 1
    X = np.stack([values[i:i + T] for i in range(n)])                    # (n,T,N,F)
    TC = np.stack([tcodes[i:i + T] for i in range(n)])                   # (n,T,3)
    Y = np.stack([values[i + T:i + T + H, target_idx, aqi_col] for i in range(n)])  # (n,H)
    return X, TC, Y


def prepare(cfg, city, T, H):
    values, coords, ts, target_idx, aqi_col = load_city(city)
    n = len(values) - T - H + 1
    n_tr = int(n * cfg["split"]["train"])
    n_va = int(n * cfg["split"]["val"])
    # Min-max statistics from TRAIN rows only (avoids test-set leakage).
    train_rows = values[: n_tr + T + H - 1]
    vmin = train_rows.min(axis=(0, 1))                                    # (F,)
    vmax = train_rows.max(axis=(0, 1))
    scale = np.where(vmax - vmin == 0, 1.0, vmax - vmin)
    scaled = (values - vmin) / scale

    X, TC, Y = make_windows(scaled, time_codes(ts), aqi_col, target_idx, T, H)
    Y = Y * scale[aqi_col] + vmin[aqi_col]                                # keep targets in AQI units
    Y_s = (Y - vmin[aqi_col]) / scale[aqi_col]                            # scaled targets for training

    sl = {"train": slice(0, n_tr), "val": slice(n_tr, n_tr + n_va), "test": slice(n_tr + n_va, n)}
    SPLITS.mkdir(parents=True, exist_ok=True)
    with open(SPLITS / f"{city}_T{T}_H{H}.json", "w") as f:               # reproducible split record
        json.dump({k: [s.start, s.stop] for k, s in sl.items()}, f)

    data = {k: (torch.from_numpy(X[s]), torch.from_numpy(TC[s]),
                torch.from_numpy(Y_s[s].astype(np.float32)), Y[s]) for k, s in sl.items()}
    meta = dict(coords=coords, target_idx=target_idx, in_dim=X.shape[-1], num_nodes=X.shape[2],
                y_min=float(vmin[aqi_col]), y_scale=float(scale[aqi_col]), aqi_col=aqi_col,
                timestamps=ts.values, test_start=sl["test"].start)
    return data, meta


# ----------------------------------------------------------------------- train / eval
def run_epoch(model, loader, device, optimizer=None, grad_clip=None):
    train = optimizer is not None
    model.train(train)
    crit, total, count = nn.MSELoss(), 0.0, 0
    with torch.set_grad_enabled(train):
        for x, tc, y in loader:
            x, tc, y = x.to(device), tc.to(device), y.to(device)
            loss = crit(model(x, tc), y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
    return total / count


@torch.no_grad()
def predict(model, X, TC, device, y_min, y_scale, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device), TC[i:i + bs].to(device)).cpu().numpy())
    return np.concatenate(out) * y_scale + y_min                          # back to AQI units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--city", default=None)
    ap.add_argument("--config", default=str(CONFIG / "config.yaml"))
    ap.add_argument("--input_steps", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    city = args.city or cfg["city"]
    T = args.input_steps or cfg["input_steps"]
    H = args.horizon or cfg["horizon"]
    tc = cfg["train"]
    epochs, lr = args.epochs or tc["epochs"], args.lr or tc["lr"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, meta = prepare(cfg, city, T, H)
    adj = build_adjacency(meta["coords"])
    cls, group = MODELS[args.model]
    model = cls(in_dim=meta["in_dim"], num_nodes=meta["num_nodes"], target_idx=meta["target_idx"],
                horizon=H, adj=adj, hidden=cfg["model"]["hidden"], heads=cfg["model"]["heads"],
                dropout=cfg["model"]["dropout"]).to(device)
    print(f"{args.model} | {city} | T={T} H={H} | params={count_parameters(model):,} | device={device}")

    mk = lambda k, shuffle=False: DataLoader(TensorDataset(*data[k][:3]), batch_size=tc["batch_size"],
                                             shuffle=shuffle, num_workers=tc["num_workers"])
    # shuffle=True only reorders mini-batches INSIDE the training period; the split stays chronological.
    train_dl, val_dl = mk("train", True), mk("val")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])

    tag = f"{args.model}_{city}_T{T}_H{H}_s{seed}"
    ckpt_dir = MODELS_SAVED / "checkpoints"
    best_path = MODELS_SAVED / group / f"{tag}.pt"
    for p in (ckpt_dir, best_path.parent, RESULTS / "metrics", RESULTS / "predictions",
              RESULTS / "logs", RESULTS / "figures"):
        p.mkdir(parents=True, exist_ok=True)

    best_rmse, best_epoch, bad, history, t0 = float("inf"), 0, 0, [], time.time()
    for ep in range(1, epochs + 1):
        tr_loss = run_epoch(model, train_dl, device, opt, tc["grad_clip"])
        Xv, TCv, _, Yv = data["val"]
        val_rmse = compute_all(Yv, predict(model, Xv, TCv, device, meta["y_min"], meta["y_scale"]))["RMSE"]
        val_loss = run_epoch(model, val_dl, device)                       # validation MSE (scaled)
        history.append({"epoch": ep, "train_mse_scaled": tr_loss, "val_mse_scaled": val_loss,
                        "val_rmse": val_rmse})
        print(f"epoch {ep:3d} | train MSE {tr_loss:.5f} | val MSE {val_loss:.5f} | val RMSE {val_rmse:.3f}")
        if val_rmse < best_rmse:                                          # keep best weights
            best_rmse, best_epoch, bad = val_rmse, ep, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= tc["patience"]:
                print(f"early stopping at epoch {ep}")
                break
        torch.save(model.state_dict(), ckpt_dir / f"{tag}_last.pt")

    # ------------------------------------------------------------------ test
    model.load_state_dict(torch.load(best_path, map_location=device))
    Xt, TCt, _, Yt = data["test"]
    pred = predict(model, Xt, TCt, device, meta["y_min"], meta["y_scale"])
    metrics = {"overall": compute_all(Yt, pred),
               **{f"step_{h + 1}": compute_all(Yt[:, h], pred[:, h]) for h in range(H)}}
    print("TEST:", json.dumps(metrics["overall"], indent=2))

    np.savez(RESULTS / "predictions" / f"{tag}.npz", observed=Yt, predicted=pred)
    with open(RESULTS / "metrics" / f"{tag}.json", "w") as f:
        json.dump({"model": args.model, "city": city, "T": T, "H": H, "seed": seed, "lr": lr,
                   "best_val_rmse": best_rmse, "test": metrics}, f, indent=2)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULTS / "logs" / f"{tag}_history.csv", index=False)

    # ------------------------------------------------------------------ plots
    fig_dir = RESULTS / "figures"
    run_info = (f"run {datetime.now():%Y-%m-%d %H:%M} | {len(history)} epochs | "
                f"{(time.time() - t0) / 60:.1f} min")
    plot_loss_curves(hist_df, fig_dir / f"{tag}_loss.png", best_epoch=best_epoch,
                     title=f"{args.model} - {city}: training vs validation loss", run_info=run_info)
    # previous (input) AQI of the target station for each test window, back in AQI units
    past = Xt[:, :, meta["target_idx"], meta["aqi_col"]].numpy() * meta["y_scale"] + meta["y_min"]
    # timestamps of every test window: input hours [i, i+T), forecast hours [i+T, i+T+H)
    start = meta["test_start"] + np.arange(len(Xt))[:, None]
    past_times = meta["timestamps"][start + np.arange(T)]                  # (n, T)
    future_times = meta["timestamps"][start + T + np.arange(H)]            # (n, H)
    plot_prediction_windows(past, Yt, pred, fig_dir / f"{tag}_windows.png",
                            past_times=past_times, future_times=future_times,
                            title=f"{args.model} - {city}: previous, real and predicted AQI")
    plot_test_series(Yt, pred, fig_dir / f"{tag}_test_series.png", times=future_times[:, 0],
                     title=f"{args.model} - {city}: real vs predicted AQI (test set)")

    log = CONFIG / "experiment_log.csv"                                   # one row per run
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "model", "city", "T", "H", "seed", "lr", "epochs_run", "params",
                        "val_RMSE", "test_RMSE", "test_MAE", "test_IA", "test_R2", "minutes"])
        o = metrics["overall"]
        w.writerow([datetime.now().isoformat(timespec="seconds"), args.model, city, T, H, seed, lr,
                    len(history), count_parameters(model), f"{best_rmse:.4f}", f"{o['RMSE']:.4f}",
                    f"{o['MAE']:.4f}", f"{o['IA']:.4f}", f"{o['R2']:.4f}", f"{(time.time() - t0) / 60:.1f}"])


if __name__ == "__main__":
    main()