from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class RunningMeanStd:
    dim: int
    eps: float = 1e-8

    def __post_init__(self):
        self.dim = int(self.dim)
        self.eps = float(self.eps)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, X: np.ndarray) -> None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"RunningMeanStd.update expects (N,{self.dim}), got {X.shape}")
        nb = int(X.shape[0])
        if nb == 0:
            return
        xb_mean = X.mean(axis=0, dtype=np.float64)
        xb_var = X.var(axis=0, dtype=np.float64)
        xb_M2 = xb_var * nb

        if self.n == 0:
            self.n = nb
            self.mean = xb_mean
            self.M2 = xb_M2
            return

        n_a = self.n
        n_b = nb
        delta = xb_mean - self.mean
        n = n_a + n_b
        self.mean = self.mean + delta * (n_b / n)
        self.M2 = self.M2 + xb_M2 + (delta * delta) * (n_a * n_b / n)
        self.n = n

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n <= 0:
            raise ValueError("RunningMeanStd has no samples.")
        var = self.M2 / max(self.n, 1)
        std = np.sqrt(np.maximum(var, 0.0)) + self.eps
        return self.mean.astype(np.float32), std.astype(np.float32)
