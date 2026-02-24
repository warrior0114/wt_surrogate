from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def r2_per_output(y_true: np.ndarray, y_pred: np.ndarray, out_names: List[str]) -> Dict[str, float]:
    out = {}
    for j, n in enumerate(out_names):
        out[n] = r2_score(y_true[:, j], y_pred[:, j])
    return out
