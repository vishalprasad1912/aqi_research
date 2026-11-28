import argparse
import csv
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models.baselines.gat_lstm import GATLSTM
from src.models.baselines.gcn_lstm import GCNLSTM
from src.evaluation.metrics import compute_all
from src.models.baselines.lstm import LSTMModel
from src.models.model_utils import count_parameters
from src.utils.paths import CONFIG_DIR, MODELS_SAVED_DIR, PROCESSED_DIR, RESULTS_DIR, SPLITS_DIR
from src.utils.seed import set_seed
from src.visualization.model_results_test import plot_loss_curves, plot_test_series

MODELS = {
    "lstm": (LSTMModel, "baselines"),
    "gcn_lstm": (GCNLSTM, "baselines"),
    "gat_lstm": (GATLSTM, "baselines")
}

def load_station_df(station):
    data = np.load(PROCESSED_DIR / "stations_data.npz", allow_pickle=True)
    station_names = data["station_names"]
    station_to_idx = {name: idx for idx, name in enumerate(station_names)}
    all_values = np.stack([data[f"station_{i}"][:, 1:].astype(np.float32) for i in station_names], axis=0)
    all_values = np.transpose(all_values, (1, 0, 2))
    values = data[f"station_{station}"]
    ts = pd.to_datetime(values[:, 0])
    target_idx = station_to_idx[station]
    return ts, target_idx, all_values

class WindowDataset(Dataset):
    def __init__(self, values, starts, T, H, target_idx, is_lstm):
        self.values = values # (43824, 17, 24)
        self.values_tensor = torch.from_numpy(values)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.T, self.H, self.target_idx, self.is_lstm = T, H, target_idx, is_lstm

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        x = self.values_tensor[s:s + self.T] # (T, N, F)
        if self.is_lstm:
            x = x[:, self.target_idx] # (T, F) target station only
        y = self.values_tensor[s + self.T:s + self.T + self.H, self.target_idx, :2] # (H, 2) PM2.5, PM10
        return x, y

    def targets(self): # (n, H, 2), scaled
        idx = self.starts[:, None] + self.T + np.arange(self.H)
        return self.values[idx, self.target_idx, :2]


def prepare(config, station, T, H, model_type):
    ts, target_idx, all_values = load_station_df(station) # ts -> (43824,), all_values -> (43824, 17, 24)
    n = len(all_values) # 43824
    tr_end = int(n * config["split"]["train"]) # 30676
    vl_end = int(n * (config["split"]["train"] + config["split"]["val"])) # 37250
    tc = config["train"]
    s_tr, s_vl, s_te = tc.get("train_stride", 1), tc.get("val_stride", 1), tc.get("test_stride", 1)

    # standard scaling using train rows only
    tr = all_values[:tr_end] # (30676, 17, 24)
    tr_mean, tr_std = tr.mean(axis=0), tr.std(axis=0) # both (17, 24)
    tr_std = np.where(tr_std == 0, 1.0, tr_std)
    scaled = ((all_values - tr_mean) / tr_std).astype(np.float32) # (43824, 17, 24)

    is_lstm = model_type == "lstm"
    def region(lo, hi, stride):
        starts = np.arange(lo, hi - T - H + 1, stride) # series
        if len(starts) == 0:
            raise ValueError(f"Region [{lo},{hi}) too short for T={T}, H={H}")
        return WindowDataset(scaled, starts, T, H, target_idx, is_lstm)

    data = {"train": region(0, tr_end, s_tr), # (1279,)
            "val":   region(tr_end, vl_end, s_vl), # (274,)
            "test":  region(vl_end, n, s_te)} # (274,) -> if 24

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_DIR / f"{station}_T{T}_H{H}.json", "w") as f:
        json.dump({"train_end_row": tr_end, "val_end_row": vl_end,
                   "strides": [s_tr, s_vl, s_te],
                   "n_windows": {k: len(v) for k, v in data.items()}}, f)
    print("windows ->", {k: len(v) for k, v in data.items()})

    meta = dict(
        target_idx=target_idx, in_dim=scaled.shape[-1], num_nodes=scaled.shape[1],
        y_mean=tr_mean[target_idx, :2], y_std=tr_std[target_idx, :2],
        timestamps=np.asarray(ts), test_starts=data["test"].starts, # array
        test_stride=s_te)
    return data, meta

def run_epoch(model, loader, device, optimizer=None, grad_clip=None, collect=False):
    train = optimizer is not None
    model.train(train)
    crit, total, count, preds = nn.MSELoss(), 0.0, 0, []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device) # x-> [B, T, N, F], y-> [B, H, 2]
            out = model(x) # [B, H, 2]
            loss = crit(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
            if collect:
                preds.append(out.detach().cpu().numpy())
    loss = total / count
    return (loss, np.concatenate(preds)) if collect else loss

@torch.no_grad()
def predict(model, X, device, y_mean=0, y_std=1, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device)).cpu().numpy())
    return np.concatenate(out) * y_std + y_mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--station", default=None)
    parser.add_argument("--config", default=str(CONFIG_DIR / "config.yaml"))
    parser.add_argument("--input_steps", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gcn_layers", type=int, default=None)
    parser.add_argument("--gat_layers", type=int, default=None)
    parser.add_argument("--lstm_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    station = args.station or config["station"]
    T = args.input_steps or config["input_steps"]
    H = args.horizon or config["horizon"]
    seed = args.seed if args.seed is not None else config["seed"]
    dropout = args.dropout if args.dropout != None else config["model"]["dropout"]
    tc = config["train"]
    lstm_layers = args.lstm_layers or tc["lstm_layers"]
    gcn_layers = args.gcn_layers or tc["gcn_layers"] if args.model == "gcn_lstm" else 0
    gat_layers = args.gat_layers or tc["gat_layers"] if args.model == "gat_lstm" else 0
    epochs, lr = args.epochs or tc["epochs"], args.lr or tc["lr"]

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, meta = prepare(config, station, T, H, str(args.model))

    cls, group = MODELS[args.model]
    if args.model == "gcn_lstm":
        model = cls(in_dim=meta["in_dim"], gcn_layers=gcn_layers, lstm_layers=lstm_layers, horizon=H,
                    hidden=config["model"]["hidden"], target_idx=meta["target_idx"], dropout=dropout).to(device)
        print(f"{args.model} | {station} | T={T} H={H} | gcn_layers={gcn_layers} | lstm_layers={lstm_layers} | "
              f"params={count_parameters(model):,} | device={device}")
        tag = f"{args.model}_{station}_T{T}_H{H}_gcn_l{gcn_layers}_lstm_l{lstm_layers}_s{seed}"
    elif args.model == "gat_lstm":
        model = cls(in_dim=meta["in_dim"], gat_layers=gat_layers, lstm_layers=lstm_layers, horizon=H,
                    hidden=config["model"]["hidden"], target_idx=meta["target_idx"], dropout=dropout).to(device)
        print(f"{args.model} | {station} | T={T} H={H} | gat_layers={gat_layers} | lstm_layers={lstm_layers} | "
            f"params={count_parameters(model):,} | device={device}")
        tag = f"{args.model}_{station}_T{T}_H{H}_gat_l{gat_layers}_lstm_l{lstm_layers}_s{seed}"
    else:
        model = cls(in_dim=meta["in_dim"], lstm_layers=lstm_layers, horizon=H,
                    hidden=config["model"]["hidden"], dropout=dropout).to(device)
        print(f"{args.model} | {station} | T={T} H={H} | lstm_layers={lstm_layers} | "
              f"params={count_parameters(model):,} | device={device}")
        tag = f"{args.model}_{station}_T{T}_H{H}_lstm_l{lstm_layers}_s{seed}"

    def make_dl(ds, shuffle, nw):
        return DataLoader(ds, batch_size=tc["batch_size"], shuffle=shuffle, num_workers=nw, persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None)

    train_dl = make_dl(data["train"], True, tc["num_workers"])
    val_dl = make_dl(data["val"], False, 0)
    test_dl = make_dl(data["test"], False, 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])

    checkpoint_dir = MODELS_SAVED_DIR / "checkpoints"
    best_model_path = checkpoint_dir / group / f"{tag}.pt"
    fig_dir = RESULTS_DIR / "figures"

    for p in (checkpoint_dir, best_model_path.parent, RESULTS_DIR / "metrics", RESULTS_DIR / "predictions", RESULTS_DIR / "logs", fig_dir):
        p.mkdir(parents=True, exist_ok=True)

    best_rmse, bad, history, best_epoch, t0 = float("inf"), 0, [], 0, time.time()
    report_val_rmse, report_val_mae, report_val_r2 = 0.0, 0.0, 0.0
    Y_val_real = data["val"].targets() * meta["y_std"] + meta["y_mean"]

    pbar = tqdm(range(1, epochs + 1), desc=f"{args.model}-{station}", unit="epoch")
    for epoch in pbar:
        train_loss = run_epoch(model, train_dl, device, optimizer, tc["grad_clip"])
        val_loss, val_pred = run_epoch(model, val_dl, device, collect=True)
        val_pred = val_pred * meta["y_std"] + meta["y_mean"]
        vm = compute_all(Y_val_real, val_pred)
        val_rmse, val_mae, val_r2 = vm["RMSE"], vm["MAE"], vm["R2"]

        history.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_mse_scaled": val_loss, "val_rmse": val_rmse, "val_mae": val_mae})
        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                         rmse=f"{val_rmse:.2f}", best=f"{min(best_rmse, val_rmse):.2f}")
        tqdm.write(f"epoch {epoch:3d} | train MSE (scaled) {train_loss:.4f} | val MSE (scaled) {val_loss:.4f}"
                   f" | val RMSE {val_rmse:.3f} | val MAE {val_mae:.3f}")

        torch.save(model.state_dict(), checkpoint_dir / f"{tag}_last.pt")
        if val_rmse < best_rmse:
            best_rmse, best_epoch, bad = val_rmse, epoch, 0
            report_val_rmse, report_val_mae, report_val_r2 = val_rmse, val_mae, val_r2
            torch.save(model.state_dict(), best_model_path)
        else:
            bad += 1
            if bad >= tc["patience"]:
                tqdm.write(f"early stopping at epoch {epoch}")
                break
    pbar.close()

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, pred = run_epoch(model, test_dl, device, collect=True)
    pred = pred * meta["y_std"] + meta["y_mean"]
    Y_test_real = data["test"].targets() * meta["y_std"] + meta["y_mean"]

    future_times = meta["timestamps"][meta["test_starts"][:, None] + T + np.arange(H)]  # (n_test, H)

    metrics = {"overall": compute_all(Y_test_real, pred),
               **{f"step_{h + 1}": compute_all(Y_test_real[:, h], pred[:, h]) for h in range(H)},
               **{f"target_feature_{k + 1}": compute_all(Y_test_real[:, :, k], pred[:, :, k]) for k in range(2)}}

    with open(RESULTS_DIR / "metrics" / f"{tag}.json", "w") as f:
        json.dump({"model": args.model, "station": station, "T": T, "H": H,
                   "gat_layers": gat_layers, "gcn_layers": gcn_layers, "lstm_layers": lstm_layers,
                   "seed": seed, "lr": lr, "best_val_rmse": best_rmse,
                   "strides": [tc.get("train_stride", 1), tc.get("val_stride", 1), meta["test_stride"]],
                   "test": metrics}, f, indent=2)

    step = max(1, H // meta["test_stride"])
    sel = slice(None, None, step)
    df = pd.DataFrame({
        "Time": pd.to_datetime(future_times[sel].reshape(-1)),
        "PM2.5_Actual": Y_test_real[sel, :, 0].reshape(-1), "PM2.5_Pred": pred[sel, :, 0].reshape(-1),
        "PM10_Actual": Y_test_real[sel, :, 1].reshape(-1), "PM10_Pred": pred[sel, :, 1].reshape(-1),
    })
    df.to_csv(RESULTS_DIR / "predictions" / f"{tag}.csv", index=False)

    plot_test_series(Y_test_real, pred, fig_dir / f"{tag}_test_series.png", times=future_times,
                     stride=meta["test_stride"], title=f"{args.model} - {station}: real vs predicted")
    plot_test_series(Y_test_real, pred, fig_dir / f"{tag}_test_step1.png", times=future_times,
                     horizon_step=0, title=f"{args.model} - {station}: 1-step ahead")
    plot_test_series(Y_test_real, pred, fig_dir / f"{tag}_test_stepH.png", times=future_times,
                     horizon_step=H - 1, title=f"{args.model} - {station}: {H}-step ahead")

    print("TEST:", json.dumps(metrics["overall"], indent=2))

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULTS_DIR / "logs" / f"{tag}_history.csv", index=False)
    plot_loss_curves(hist_df, fig_dir / f"{tag}_loss.png", best_epoch=best_epoch,
                     title=f"{args.model} - {station}: training vs validation loss")

    log = CONFIG_DIR / "experiment_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "model", "station", "T", "H", "seed", "lr", "epochs_run", "dropout",
                        "gat_layers", "gcn_layers", "lstm_layers",
                        "val_RMSE", "val_MAE", "val_R2", "test_RMSE", "test_MAE", "test_R2", "minutes"])
        o = metrics["overall"]
        w.writerow([datetime.now().isoformat(timespec="seconds"), args.model, station, T, H, seed, lr,
                    len(history), dropout, gat_layers, gcn_layers, lstm_layers,
                    f"{report_val_rmse:.4f}", f"{report_val_mae:.4f}",
                    f"{report_val_r2:.4f}", f"{o['RMSE']:.4f}",
                    f"{o['MAE']:.4f}", f"{o['R2']:.4f}", f"{(time.time() - t0) / 60:.2f}"])

if __name__ == "__main__":
    main()