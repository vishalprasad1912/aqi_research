import numpy as np

def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))

def mae(y, p):
    return float(np.mean(np.abs(y - p)))

def ia(y, p):
    ybar = y.mean()
    num = np.sum((p - y) ** 2)
    den = np.sum((np.abs(p - ybar) + np.abs(y - ybar)) ** 2)
    return float(1.0 - num / den)

def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)

def compute_all(y, p):
    y, p = np.asarray(y, dtype=np.float64).ravel(), np.asarray(p, dtype=np.float64).ravel()
    return {"RMSE": rmse(y, p), "MAE": mae(y, p), "IA": ia(y, p), "R2": r2(y, p)}