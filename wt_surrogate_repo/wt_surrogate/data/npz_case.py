from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from wt_surrogate.utils.numpy import safe_float32
from wt_surrogate.features.otf import OTFFeatureExtractor, get_wind_features
from wt_surrogate.features.preprocess import angle_encode, add_input_derivatives, apply_history


@dataclass
class CaseProcessConfig:
    desired_input_params: List[str]
    desired_output_params: List[str]

    angle_mode: str = "harmonics6"
    keep_raw_angles: bool = False

    history_k: int = 0
    history_on: str = "both"  # both or wind-only

    use_all_features: bool = False
    kept_feature_indices: Optional[List[int]] = None

    add_input_derivatives: bool = False
    deriv_order: int = 1
    dt: float = 0.1

    lag_steps: int = 0


def process_npz_case(fp: str, cfg: CaseProcessConfig, otf: Optional[OTFFeatureExtractor] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    data = np.load(fp, allow_pickle=True)
    inputs_full = safe_float32(data["input_params"])
    outputs_full = safe_float32(data["output_params"])

    input_names_all = [str(x) for x in data["input_names"].tolist()] if "input_names" in data.files else []
    output_names_all = [str(x) for x in data["output_names"].tolist()] if "output_names" in data.files else []

    wind, wind_names = get_wind_features(data, otf=otf, kept_feature_indices=cfg.kept_feature_indices, use_all_features=cfg.use_all_features)

    in_map = {n: i for i, n in enumerate(input_names_all)}
    missing_in = [n for n in cfg.desired_input_params if n not in in_map]
    if missing_in:
        raise KeyError(f"{os.path.basename(fp)} missing input columns: {missing_in}")
    inp_raw = inputs_full[:, [in_map[n] for n in cfg.desired_input_params]]
    inp_names = list(cfg.desired_input_params)

    out_map = {n: i for i, n in enumerate(output_names_all)}
    missing_out = [n for n in cfg.desired_output_params if n not in out_map]
    if missing_out:
        raise KeyError(f"{os.path.basename(fp)} missing output columns: {missing_out}")
    out = outputs_full[:, [out_map[n] for n in cfg.desired_output_params]]
    out_names = list(cfg.desired_output_params)

    if cfg.lag_steps > 0:
        if wind.shape[0] <= cfg.lag_steps:
            return wind[:0], inp_raw[:0], out[:0], wind_names, inp_names, out_names
        wind = wind[:-cfg.lag_steps]
        inp_raw = inp_raw[:-cfg.lag_steps]
        out = out[cfg.lag_steps:]

    if cfg.add_input_derivatives:
        inp_raw, inp_names = add_input_derivatives(inp_raw, inp_names, dt=cfg.dt, order=cfg.deriv_order)

    inp, inp_names = angle_encode(inp_raw, inp_names, mode=cfg.angle_mode, keep_raw_angles=cfg.keep_raw_angles)

    if cfg.history_k > 0:
        wind_h = apply_history(wind, cfg.history_k)
        if cfg.history_on == "both":
            inp_h = apply_history(inp, cfg.history_k)
        else:
            inp_h = inp[cfg.history_k:]
        out_h = out[cfg.history_k:]
        wind, inp, out = wind_h, inp_h, out_h

    return wind, inp, out, wind_names, inp_names, out_names


def count_effective_samples(fp: str, cfg: CaseProcessConfig, otf: Optional[OTFFeatureExtractor] = None) -> int:
    w, i, o, *_ = process_npz_case(fp, cfg, otf=otf)
    return int(o.shape[0])
