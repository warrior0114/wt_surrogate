from __future__ import annotations
from typing import List, Tuple
import numpy as np

def angle_encode(inputs: np.ndarray, input_names: List[str], mode: str = "harmonics6", keep_raw_angles: bool = False) -> Tuple[np.ndarray, List[str]]:
    if mode == "none":
        return inputs, input_names

    name2i = {n: i for i, n in enumerate(input_names)}
    out_cols, out_names = [], []

    def add_col(col, name):
        out_cols.append(col)
        out_names.append(name)

    for n in input_names:
        if (not keep_raw_angles) and (n in ("Azimuth", "NacYaw")):
            continue
        add_col(inputs[:, name2i[n]], n)

    def add_sincos(name: str):
        if name not in name2i:
            return
        a = inputs[:, name2i[name]] * (np.pi / 180.0)
        add_col(np.sin(a), f"{name}_sin")
        add_col(np.cos(a), f"{name}_cos")

    def add_harmonics_azimuth(include_6p: bool):
        if "Azimuth" not in name2i:
            return
        a = inputs[:, name2i["Azimuth"]] * (np.pi / 180.0)
        add_col(np.sin(a), "Azimuth_sin1")
        add_col(np.cos(a), "Azimuth_cos1")
        add_col(np.sin(3.0 * a), "Azimuth_sin3")
        add_col(np.cos(3.0 * a), "Azimuth_cos3")
        if include_6p:
            add_col(np.sin(6.0 * a), "Azimuth_sin6")
            add_col(np.cos(6.0 * a), "Azimuth_cos6")

    if mode == "sincos":
        add_sincos("Azimuth"); add_sincos("NacYaw")
    elif mode == "harmonics":
        add_harmonics_azimuth(include_6p=False); add_sincos("NacYaw")
    elif mode == "harmonics6":
        add_harmonics_azimuth(include_6p=True); add_sincos("NacYaw")
    else:
        raise ValueError(f"Unknown angle_mode: {mode}")

    X = np.stack(out_cols, axis=1).astype(np.float32, copy=False)
    return X, out_names

def add_input_derivatives(inp_raw: np.ndarray, inp_names: List[str], dt: float = 0.1, order: int = 2) -> Tuple[np.ndarray, List[str]]:
    name2i = {n: i for i, n in enumerate(inp_names)}
    targets = [n for n in ("BldPitch1", "BldPitch2", "BldPitch3", "RotSpeed") if n in name2i]
    if (not targets) or order <= 0:
        return inp_raw, inp_names

    dt = max(float(dt), 1e-6)
    cols = [inp_raw]
    names = list(inp_names)

    first = {}
    for n in targets:
        x = inp_raw[:, name2i[n]]
        dx = np.empty_like(x)
        dx[0] = 0.0
        dx[1:] = (x[1:] - x[:-1]) / dt
        first[n] = dx
        cols.append(dx[:, None]); names.append(f"d{n}")

    if order >= 2:
        for n in targets:
            dx = first[n]
            ddx = np.empty_like(dx)
            ddx[0] = 0.0
            ddx[1:] = (dx[1:] - dx[:-1]) / dt
            cols.append(ddx[:, None]); names.append(f"d2{n}")

    out = np.concatenate(cols, axis=1).astype(np.float32, copy=False)
    return out, names

def apply_history(X: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 0:
        return X
    T, D = X.shape
    if T <= k:
        return X[:0, :].reshape(0, D * (k + 1))
    chunks = [X[k - i: T - i] for i in range(0, k + 1)]
    return np.concatenate(chunks, axis=1)
