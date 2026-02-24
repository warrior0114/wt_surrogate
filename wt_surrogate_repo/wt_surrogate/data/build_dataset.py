from __future__ import annotations
import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from wt_surrogate.io.openfast_out import read_openfast_out_ascii, resample_df_to_times, compute_target_times
from wt_surrogate.io.turbsim_bts import read_bts_official


@dataclass
class BuildDatasetConfig:
    """Configuration for building NPZ datasets."""
    bts_dir: str
    openfast_out_root: str
    output_dir: str

    input_cols: Tuple[str, ...] = ("BldPitch1","BldPitch2","BldPitch3","Azimuth","RotSpeed","NacYaw")
    output_cols: Tuple[str, ...] = ("RootFxb1","RootMyb1","TwrBsFxt","TwrBsMyt","TTDspFA","GenPwr")

    time_start: float = 100.0
    time_end: float = 700.0
    time_dt: float = 0.1

    out_exts: Tuple[str, ...] = (".out",)

    store_wind: bool = True
    wind_save_dtype: str = "float16"  # float16 or float32
    store_wind_idx: bool = True


def iter_out_files(root_dir: str, out_exts: Tuple[str, ...]) -> List[str]:
    out_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in out_exts:
                out_paths.append(os.path.join(dirpath, fn))
    out_paths.sort()
    return out_paths


def build_bts_index(bts_root: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(bts_root):
        for fn in filenames:
            if fn.lower().endswith('.bts'):
                stem = os.path.splitext(fn)[0]
                p = os.path.join(dirpath, fn)
                if stem not in idx:
                    idx[stem] = p
    return idx


def make_case_id(out_path: str, root_dir: str) -> Tuple[str, str]:
    stem = os.path.splitext(os.path.basename(out_path))[0]
    rel = os.path.relpath(out_path, root_dir)
    rel_noext = os.path.splitext(rel)[0]
    case_id = rel_noext.replace(os.sep, '__')
    return case_id, stem


def build_npz_dataset(cfg: BuildDatasetConfig) -> str:
    """Build per-case NPZ files and a manifest.csv.

    Returns path to manifest.csv.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    manifest_path = os.path.join(cfg.output_dir, "manifest.csv")

    out_paths = iter_out_files(cfg.openfast_out_root, cfg.out_exts)
    if not out_paths:
        raise FileNotFoundError(f"No OpenFAST outputs found under {cfg.openfast_out_root} with {cfg.out_exts}")

    bts_index = build_bts_index(cfg.bts_dir)

    skip_stats = {
        "bts_missing": 0,
        "out_read_error": 0,
        "missing_required_cols": 0,
        "bts_read_error": 0,
        "insufficient_time_range": 0,
    }
    manifest_rows: List[Dict[str, object]] = []
    total_saved = 0

    for out_path in tqdm(out_paths, desc="Build NPZ", ncols=100):
        case_id, stem = make_case_id(out_path, cfg.openfast_out_root)

        bts_path = os.path.join(cfg.bts_dir, stem + '.bts')
        if not os.path.exists(bts_path):
            bts_path = bts_index.get(stem, "")
        if not bts_path or (not os.path.exists(bts_path)):
            skip_stats["bts_missing"] += 1
            continue

        try:
            df, units_map = read_openfast_out_ascii(out_path)
        except Exception:
            skip_stats["out_read_error"] += 1
            continue

        required_cols = ['Time', *cfg.input_cols, *cfg.output_cols]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            skip_stats["missing_required_cols"] += 1
            continue

        try:
            header, wind_data = read_bts_official(bts_path)
        except Exception:
            skip_stats["bts_read_error"] += 1
            continue
        if header is None or wind_data is None:
            skip_stats["bts_read_error"] += 1
            continue

        bts_dt = float(header.get('dt', 0.05))
        bts_nt = int(wind_data.shape[0])
        out_time_max = float(df['Time'].max())

        target_times = compute_target_times(cfg.time_start, cfg.time_end, cfg.time_dt, out_time_max, bts_dt, bts_nt)
        if target_times is None or len(target_times) < 2:
            skip_stats["insufficient_time_range"] += 1
            continue

        input_params = resample_df_to_times(df, target_times, list(cfg.input_cols))
        output_params = resample_df_to_times(df, target_times, list(cfg.output_cols))

        idx = np.round(target_times / bts_dt).astype(np.int64)
        idx = np.clip(idx, 0, bts_nt - 1)
        wind_selected = wind_data[idx]  # (N,3,ny,nz)

        wind_to_save = None
        if cfg.store_wind:
            if str(cfg.wind_save_dtype).lower() == 'float16':
                wind_to_save = wind_selected.astype(np.float16, copy=False)
            else:
                wind_to_save = wind_selected.astype(np.float32, copy=False)

        out_fp = os.path.join(cfg.output_dir, f"{case_id}_data.npz")
        input_names = np.array(cfg.input_cols, dtype='U')
        output_names = np.array(cfg.output_cols, dtype='U')
        input_units = np.array([units_map.get(c, '') for c in cfg.input_cols], dtype='U')
        output_units = np.array([units_map.get(c, '') for c in cfg.output_cols], dtype='U')

        np.savez_compressed(
            out_fp,
            case_id=np.array(case_id, dtype='U'),
            out_path=np.array(out_path, dtype='U'),
            bts_path=np.array(bts_path, dtype='U'),
            npz_dir=np.array(os.path.dirname(out_fp), dtype='U'),
            time=target_times.astype(np.float32),

            input_params=input_params,
            input_names=input_names,
            input_units=input_units,

            output_params=output_params,
            output_names=output_names,
            output_units=output_units,

            bts_dt=np.float32(bts_dt),
            bts_grid=np.array([int(header.get('ny', -1)), int(header.get('nz', -1))], dtype=np.int32),

            wind_dtype=np.array(str(cfg.wind_save_dtype), dtype='U'),
            store_wind=np.int32(1 if cfg.store_wind else 0),
            wind=wind_to_save if cfg.store_wind else None,
            wind_idx=idx.astype(np.int64) if cfg.store_wind_idx else None,
        )

        total_saved += 1
        manifest_rows.append({
            "case_id": case_id,
            "stem": stem,
            "out_path": out_path,
            "bts_path": bts_path,
            "n_samples": int(output_params.shape[0]),
            "t_start": float(target_times[0]),
            "t_end": float(target_times[-1]),
            "bts_dt": float(bts_dt),
            "grid_in": f"{wind_selected.shape[-2]}x{wind_selected.shape[-1]}",
            "wind_saved": int(cfg.store_wind),
            "wind_dtype": str(cfg.wind_save_dtype),
            "npz_path": out_fp,
        })

    if manifest_rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
    else:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            f.write("")

    return manifest_path
