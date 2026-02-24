from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from wt_surrogate.io.turbsim_bts import read_bts_official
from wt_surrogate.features.PIFENet import PIFENet


def _npz_scalar_to_str(x) -> Optional[str]:
    if x is None:
        return None
    try:
        if isinstance(x, np.ndarray) and x.shape == ():
            return str(x.item())
        return str(x)
    except Exception:
        return None


def resolve_bts_path(npz: np.lib.npyio.NpzFile, bts_root: Optional[str] = None) -> str:
    if 'bts_path' not in npz.files and 'bts_path_rel' not in npz.files:
        raise KeyError("NPZ missing bts_path/bts_path_rel")

    p_abs = _npz_scalar_to_str(npz['bts_path']) if 'bts_path' in npz.files else None
    p_rel = _npz_scalar_to_str(npz['bts_path_rel']) if 'bts_path_rel' in npz.files else None
    npz_dir = _npz_scalar_to_str(npz['npz_dir']) if 'npz_dir' in npz.files else None

    cand = []
    if p_abs:
        cand.append(p_abs)
    if p_abs and (not os.path.isabs(p_abs)) and npz_dir:
        cand.append(os.path.join(npz_dir, p_abs))
    if bts_root and p_rel:
        cand.append(os.path.join(bts_root, p_rel))
    if p_rel and npz_dir:
        cand.append(os.path.join(npz_dir, p_rel))
    if p_rel:
        cand.append(p_rel)

    for c in cand:
        if c and os.path.exists(c):
            return c
    return cand[0] if cand else (p_abs or p_rel or "")


def load_wind_snapshots(npz: np.lib.npyio.NpzFile, row_idx: Optional[np.ndarray] = None, bts_root: Optional[str] = None) -> np.ndarray:
    if 'wind' in npz.files and npz['wind'] is not None:
        wind_all = npz['wind']
        if wind_all.ndim != 4:
            raise ValueError(f"NPZ wind expects 4D (T,3,ny,nz), got {wind_all.shape}")
        if row_idx is None:
            return wind_all.astype(np.float32, copy=False)
        return wind_all[row_idx].astype(np.float32, copy=False)

    bts_path = resolve_bts_path(npz, bts_root=bts_root)
    header, wind_data = read_bts_official(bts_path)
    if wind_data is None:
        raise ValueError(f"Failed to read BTS: {bts_path}")

    if 'wind_idx' in npz.files and npz['wind_idx'] is not None:
        idx_all = npz['wind_idx'].astype(np.int64)
    else:
        if 'time' not in npz.files or 'bts_dt' not in npz.files:
            raise KeyError('NPZ missing wind_idx and time/bts_dt for alignment')
        t = npz['time'].astype(np.float64)
        bts_dt = float(npz['bts_dt'])
        idx_all = np.clip(np.round(t / bts_dt).astype(np.int64), 0, wind_data.shape[0] - 1)

    if row_idx is None:
        return wind_data[idx_all].astype(np.float32, copy=False)
    return wind_data[idx_all[row_idx]].astype(np.float32, copy=False)


@dataclass
class OTFFeatureConfig:
    weights_dir: str
    device: str = "cpu"
    batch: int = 2048
    bts_root: Optional[str] = None


class OTFFeatureExtractor:
    def __init__(self, cfg: OTFFeatureConfig):
        self.cfg = cfg
        self._extractor: Optional[torch.nn.Module] = None
        self._grid: Optional[Tuple[int, int]] = None

    def _get_extractor(self) -> torch.nn.Module:
        if self._extractor is not None:
            return self._extractor
        if not self.cfg.weights_dir or (not os.path.exists(self.cfg.weights_dir)):
            raise FileNotFoundError(f"weights_dir does not exist: {self.cfg.weights_dir}")
        dev = torch.device(self.cfg.device)
        ex = PIFENet(weights_dir=self.cfg.weights_dir).to(dev)
        ex.eval()
        try:
            self._grid = (int(ex.hub.shape[-2]), int(ex.hub.shape[-1]))
        except Exception:
            self._grid = None
        self._extractor = ex
        return ex

    @property
    def feature_names(self) -> List[str]:
        ex = self._get_extractor()
        names = getattr(ex, 'feature_names', None)
        return [str(x) for x in list(names)] if names is not None else []

    def extract(self, npz: np.lib.npyio.NpzFile, row_idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        ex = self._get_extractor()
        wind = load_wind_snapshots(npz, row_idx=row_idx, bts_root=self.cfg.bts_root)
        if wind.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32), self.feature_names

        dev = torch.device(self.cfg.device)
        feats_list: List[np.ndarray] = []
        bs = max(1, int(self.cfg.batch))
        with torch.no_grad():
            for s in range(0, wind.shape[0], bs):
                w = torch.from_numpy(wind[s:s+bs]).to(dev)
                if self._grid is not None and ((w.shape[-2] != self._grid[0]) or (w.shape[-1] != self._grid[1])):
                    w = F.interpolate(w, size=self._grid, mode='bilinear', align_corners=False)
                f = ex(w).detach().cpu().numpy().astype(np.float32, copy=False)
                feats_list.append(f)

        feats = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, 0), dtype=np.float32)
        names = self.feature_names or [f"f{i}" for i in range(feats.shape[1])]
        return feats, names


def get_wind_features(npz: np.lib.npyio.NpzFile, otf: Optional[OTFFeatureExtractor], kept_feature_indices: Optional[List[int]] = None, use_all_features: bool = False) -> Tuple[np.ndarray, List[str]]:
    if 'features' in npz.files:
        feats = npz['features'].astype(np.float32, copy=False)
        names = [str(x) for x in npz['feature_names'].tolist()] if 'feature_names' in npz.files else [f"feat_{i}" for i in range(feats.shape[1])]
    else:
        if otf is None:
            raise ValueError("NPZ has no precomputed 'features' and no OTFFeatureExtractor was provided.")
        feats, names = otf.extract(npz, row_idx=None)

    if use_all_features or kept_feature_indices is None:
        return feats, names
    idx = list(kept_feature_indices)
    return feats[:, idx], [names[i] for i in idx]
