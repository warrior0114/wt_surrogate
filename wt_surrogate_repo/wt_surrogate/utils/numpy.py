import numpy as np

def safe_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x.astype(np.float32, copy=False) if x.dtype != np.float32 else x
