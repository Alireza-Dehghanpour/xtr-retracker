import numpy as np


def mse_np(a, b):
    return float(np.mean((a - b) ** 2))


def rmse_np(a, b):
    return float(np.sqrt(mse_np(a, b)))


def mae_np(a, b):
    return float(np.mean(np.abs(a - b)))


def r2_np(a, b):
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-12))