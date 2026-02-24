from __future__ import annotations
from io import StringIO
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def make_unique(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
    return out

def read_openfast_out_ascii(out_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    with open(out_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Time'):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Cannot find header line starting with 'Time' in: {out_path}")

    raw_names = lines[header_idx].strip().split()
    names = make_unique(raw_names)

    units_map: Dict[str, str] = {}
    if header_idx + 1 < len(lines):
        unit_tokens = lines[header_idx + 1].strip().split()
        if len(unit_tokens) == len(names):
            units_map = {names[j]: unit_tokens[j] for j in range(len(names))}

    data_lines = lines[header_idx + 2:]
    df = pd.read_csv(
        StringIO(''.join(data_lines)),
        delim_whitespace=True,
        names=names,
        engine='python'
    )

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['Time']).sort_values('Time')
    df = df.drop_duplicates(subset=['Time'], keep='first').reset_index(drop=True)
    return df, units_map

def resample_df_to_times(df: pd.DataFrame, target_times: np.ndarray, cols: List[str]) -> np.ndarray:
    src_t = df['Time'].to_numpy(dtype=np.float64)
    out = np.zeros((len(target_times), len(cols)), dtype=np.float32)
    for j, col in enumerate(cols):
        y = df[col].to_numpy(dtype=np.float64)
        out[:, j] = np.interp(target_times, src_t, y).astype(np.float32)
    return out

def compute_target_times(time_start: float, time_end: float, dt: float, out_time_max: float, bts_dt: float, bts_nt: int) -> Optional[np.ndarray]:
    bts_time_max = (bts_nt - 1) * bts_dt
    end = min(time_end, out_time_max, bts_time_max)
    if end < time_start:
        return None
    n = int(np.floor((end - time_start) / dt))
    last = time_start + n * dt
    return np.arange(time_start, last + 1e-9, dt, dtype=np.float64)
