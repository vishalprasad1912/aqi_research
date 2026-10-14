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

from src.models.baselines.gcn_lstm import GCNLSTM
from src.evaluation.metrics import compute_all
from src.utils.paths import VISUALIZATION_DIR, EXTERNAL_DIR
from src.models.baselines.lstm import LSTMModel
from src.models.model_utils import count_parameters
from src.utils.paths import CONFIG_DIR, MODELS_SAVED_DIR, PROCESSED_DIR, RESULTS_DIR, SPLITS_DIR
from src.utils.seed import set_seed
from src.visualization.model_results_test import plot_loss_curves, plot_test_series

MODELS = {
    "lstm": (LSTMModel, "baselines"),
    "gcn_lstm": (GCNLSTM, "baselines"),
    # "glstm_block": (GLSTMBlockModel, "proposed"),
}

# def load_station_df(station, date_col="Timestamp"):
#     df = pd.read_csv(PROCESSED_DIR / f"{station}")
#     df[date_col] = pd.to_datetime(df[date_col])
#     df = df.set_index(date_col)
#     float_cols = df.select_dtypes(include=["float64"]).columns
#     df[float_cols] = df[float_cols].astype("float32")
#     target_index = 0
#     return df, df.index, target_index

def load_station_df(station):
    data = np.load(PROCESSED_DIR / "stations_data.npz", allow_pickle=True)
    station_names = data["station_names"]

    station_to_idx = {
        name: idx
        for idx, name in enumerate(station_names)
    }
    all_values = np.stack(
        [data[f"station_{i}"][:, 1:] for i in station_names],
        axis=0
    )
    all_values = np.transpose(all_values, (1, 0, 2))
    values = data[f"station_{station}"]
    target_index = station_to_idx[station]
    return values[:, 1:], values[:, 0], target_index, all_values

def time_codes(ts):
    # (T_total, 3) long: hour 0-23, weekday 0-6, month 0-11
    return np.stack([ts.hour, ts.dayofweek, ts.month - 1], axis=1).astype(np.int32)

def make_windows(values, T, H, tcodes, features_idx=[0, 1]):
    # input rows [i, i+T), target rows [i+T, i+T+H)
    instances = len(values) - T - H + 1
    X = np.stack([values[i:i + T] for i in range(instances)]) # (instances, T, N, F)
    Y = np.stack([values[i + T:i + T + H, features_idx] for i in range(instances)])  # (instances, H)
    TC = np.stack([tcodes[i : i+T] for i in range(instances)]) # (instances, T, 3)
    return X, Y, TC

def prepare(config, station, T, H):
    values, ts, target_idx, all_values = load_station_df(station)
    instances = values.shape[0] - T - H + 1
    tr_instances = int(instances * config["split"]["train"])
    vl_instances = int(instances * config["split"]["val"])

    train_rows = values[: tr_instances + T + H - 1]
    vmin = np.min(train_rows, axis=0)
    vmax = np.max(train_rows, axis=0)
    scale = np.where(vmax - vmin == 0, 1.0, vmax - vmin)
    scaled = (values - vmin) / scale

    X, Y, TC = make_windows(scaled, T, H, time_codes(ts))

    sl = {
        "train": slice(0, tr_instances),
        "val": slice(tr_instances, tr_instances + vl_instances),
        "test": slice(tr_instances + vl_instances, instances)
    }

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_DIR / f"{station}_T{T}_H{H}.json", "w") as f:
        json.dump({k: [s.start, s.stop] for k, s in sl.items()}, f)

    data = {k: (torch.from_numpy(X[s]), torch.from_numpy(Y[s]), torch.from_numpy(TC[s])) for k, s in sl.items()}
    meta = dict(target_idx=target_idx, in_dim=X.shape[-1], num_nodes=X.shape[2], y_min=vmin[target_idx],
                y_scale=scale[target_idx], timestamps=np.asarray(ts), test_start=sl["test"].start)
    return data, meta

# model train and eval
def run_epoch(model, loader, device, optimizer=None, grad_clip=None):
    train = optimizer is not None
    model.train(train)
    criterion, total_loss, instance_count = nn.MSELoss(), 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y, tc in loader:
            x, y, tc = x.to(device), y.to(device), tc.to(device) # x-> [B, T, N, F], y-> [B, T, N, 1], tc-> [B, T, 3]
            # batch_pred = model(x, tc)
            batch_pred = model(x)
            batch_loss = criterion(y, batch_pred)
            if train:
                optimizer.zero_grad()
                batch_loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += batch_loss.item() * len(x)
            instance_count += len(x)
    return total_loss / instance_count

@torch.no_grad()
def predict(model, X, TC, device, y_min=0, y_scale=0, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        # out.append(model(X[i:i + bs].to(device), TC[i:i + bs].to(device)).cpu().numpy())
        out.append(model(X[i:i + bs].to(device)).cpu().numpy())
    return np.concatenate(out) * y_scale + y_min

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
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    station = args.station or config["station"]
    T = args.input_steps or config["input_steps"]
    H = args.horizon or config["horizon"]
    tc = config["train"]
    epochs, lr = args.epochs or tc["epochs"], args.lr or tc["lr"]
    seed = args.seed if args.seed is not None else config["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, meta = prepare(config, station, T, H)
    adj = np.load(EXTERNAL_DIR / "adjacency_matrix.npy") if args.model != "lstm" else None
    cls, group = MODELS[args.model]
    if args.model == "lstm":
        model = cls(in_dim=meta["in_dim"], horizon=H, hidden=config["model"]["hidden"],
                    dropout=config["model"]["dropout"]).to(device)
    else:
        model = cls(in_dim=meta["in_dim"], target_idx=meta["target_idx"],
                    horizon=H, adj=adj, hidden=config["model"]["hidden"],
                    dropout=config["model"]["dropout"]).to(device)
    # model = cls(in_dim=meta["in_dim"], target_idx=meta["target_idx"],
    #             horizon=H, hidden=config["model"]["hidden"], dropout=config["model"]["dropout"]).to(device)
    print(f"{args.model} | {station} | T={T} H={H} | params={count_parameters(model):,} | device={device}")

    mk = lambda k: DataLoader(TensorDataset(*data[k][:3]), batch_size=tc["batch_size"],
                                             shuffle=False, num_workers=tc["num_workers"])

    train_dl, val_dl = mk("train"), mk("val")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])

    tag = f"{args.model}_{station}_T{T}_H{H}_s{seed}"
    checkpoint_dir = MODELS_SAVED_DIR / "checkpoints"
    best_model_path = checkpoint_dir / group / f"{tag}.pt"
    fig_dir = RESULTS_DIR / "figures"

    for p in (checkpoint_dir, best_model_path.parent, RESULTS_DIR / "metrics", RESULTS_DIR / "predictions", RESULTS_DIR / "logs", fig_dir):
        p.mkdir(parents=True, exist_ok=True)

    best_rmse, bad, history, best_epoch, t0 = float("inf"), 0, [], 0, time.time()
    report_val_rmse, report_val_mae, report_val_r2 = 0.0, 0.0, 0.0
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_dl, device, optimizer, tc["grad_clip"])
        X_val, Y_val, TC_val = data["val"]

        val_loss = run_epoch(model, val_dl, device)

        val_rmse = compute_all((Y_val.cpu().numpy() * meta["y_scale"]) + meta["y_min"],
                               predict(model, X_val, TC_val, device, meta["y_min"], meta["y_scale"]))["RMSE"]
        val_mae = compute_all((Y_val.cpu().numpy() * meta["y_scale"]) + meta["y_min"],
                               predict(model, X_val, TC_val, device, meta["y_min"], meta["y_scale"]))["MAE"]
        val_r2 = compute_all((Y_val.cpu().numpy() * meta["y_scale"]) + meta["y_min"],
                              predict(model, X_val, TC_val, device, meta["y_min"], meta["y_scale"]))["R2"]

        history.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_mse_scaled": val_loss,
                        "val_rmse": val_rmse, "val_mae": val_mae})
        print(f"epoch {epoch:3d} | train MSE (scaled) {train_loss:.3f} | val MSE (scaled) {val_loss:.3f}"
              f" | val RMSE {val_rmse:.3f} | val MAE {val_mae:.3f}")
        if val_rmse < best_rmse:
            best_rmse, best_epoch, bad = val_rmse, epoch, 0
            report_val_rmse, report_val_mae, report_val_r2 = val_rmse, val_mae, val_r2
            torch.save(model.state_dict(), best_model_path)
        else:
            bad += 1
            if bad >= tc["patience"]:
                print(f"early stopping at epoch {epoch}")
                break
        torch.save(model.state_dict(), checkpoint_dir / f"{tag}_last.pt")

    # test
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    X_test, Y_test, TC_test = data["test"]
    pred = predict(model, X_test, TC_test, device, meta["y_min"], meta["y_scale"])
    Y_test_scaled = (Y_test.cpu().numpy() * meta["y_scale"]) + meta["y_min"]

    metrics = {"overall": compute_all(Y_test_scaled, pred),
               **{f"step_{h + 1}": compute_all(Y_test_scaled[:, h, :], pred[:, h, :]) for h in range(H)},
               ** {f"target_feature_{t + 1}": compute_all(Y_test_scaled[:, :, t], pred[:, :, t]) for t in range(Y_test.shape[2])}}
    print("TEST:", json.dumps(metrics["overall"], indent=2))

    df = pd.DataFrame()

    for h in range(Y_test_scaled.shape[1]):
        df[f"PM2.5_Actual_H{h + 1}"] = Y_test_scaled[:, h, 0]
        df[f"PM2.5_Pred_H{h + 1}"] = pred[:, h, 0]

        df[f"PM10_Actual_H{h + 1}"] = Y_test_scaled[:, h, 1]
        df[f"PM10_Pred_H{h + 1}"] = pred[:, h, 1]

    start = meta["test_start"] + np.arange(len(X_test))[:, None]

    df["Time"] = pd.to_datetime(meta["timestamps"][start + config['input_steps'] + np.arange(config['horizon'])])
    df["Time"] = pd.to_datetime(df["Time"].apply(lambda x: x[0]))
    df.set_index("Time", inplace=True)
    df.to_csv(RESULTS_DIR / "predictions" / f"{tag}", index=True)

    with open(RESULTS_DIR / "metrics" / f"{tag}.json", "w") as f:
        json.dump({"model": args.model, "station": station, "T": T, "H": H, "seed": seed, "lr": lr,
                   "best_val_rmse": best_rmse, "test": metrics}, f, indent=2)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULTS_DIR / "logs" / f"{tag}_history.csv", index=False)

    # plots
    start = meta["test_start"] + np.arange(len(X_test))[:, None]
    # past_times = meta["timestamps"][start + np.arange(T)]
    future_times = meta["timestamps"][start + T + np.arange(H)]

    plot_loss_curves(hist_df, fig_dir / f"{tag}_loss.png", best_epoch=best_epoch,
                     title=f"{args.model} - {station}: training vs validation loss")
    past_scaled = np.asarray(X_test[:, :, meta["target_idx"]].cpu().numpy() * meta["y_scale"] + meta["y_min"])
    # plot_prediction_windows(past_scaled, Y_test_scaled, pred, fig_dir / f"{tag}_windows.png", past_times = past_times, future_times = future_times,
    #                         title=f"{args.model} - {station}: previous, real and predicted AQI")
    plot_test_series(Y_test_scaled, pred, fig_dir / f"{tag}_test_series.png", times=future_times[:, 0],
                     title=f"{args.model} - {station}: real vs predicted AQI (test set)")

    log = CONFIG_DIR / "experiment_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "model", "city", "T", "H", "seed", "lr", "epochs_run", "params",
                        "val_RMSE", "val_MAE", "val_R2", "test_RMSE", "test_MAE", "test_IA", "test_R2", "minutes"])
        o = metrics["overall"]
        w.writerow([datetime.now().isoformat(timespec="seconds"), args.model, station, T, H, seed, lr,
                    len(history), count_parameters(model), f"{report_val_rmse:.4f}", f"{report_val_mae:.4f}",
                    f"{report_val_r2:.4f}", f"{o['RMSE']:.4f}",
                    f"{o['MAE']:.4f}", f"{o['IA']:.4f}", f"{o['R2']:.4f}", f"{(time.time() - t0) / 60:.2f}"])


if __name__ == "__main__":
    main()