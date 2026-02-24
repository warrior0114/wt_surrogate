from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from wt_surrogate.data.npz_case import CaseProcessConfig, process_npz_case
from wt_surrogate.data.split import stratified_split_files, simple_ratio_split, fix_empty_splits
from wt_surrogate.features.otf import OTFFeatureExtractor, OTFFeatureConfig
from wt_surrogate.models.dynamic_mlp import DynamicMLPWindTurbineModel
from wt_surrogate.training.scalers import RunningMeanStd
from wt_surrogate.training.metrics import r2_per_output, r2_score


DEFAULT_TASK_GROUPS = {
    "root": ["RootFxb1", "RootMyb1"],
    "tower": ["TwrBsFxt", "TwrBsMyt"],
    "dsp": ["TTDspFA"],
    "pwr": ["GenPwr"],
}


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str

    # OTF features
    weights_dir: Optional[str] = None
    bts_root: Optional[str] = None

    # channels
    desired_input_params: Tuple[str, ...] = ("BldPitch1","BldPitch2","BldPitch3","Azimuth","RotSpeed","NacYaw")
    desired_output_params: Tuple[str, ...] = ("RootFxb1","RootMyb1","TwrBsFxt","TwrBsMyt","TTDspFA","GenPwr")

    # feature selection
    kept_feature_indices: Optional[List[int]] = None
    use_all_features: bool = False

    # preprocessing
    angle_mode: str = "harmonics6"
    keep_raw_angles: bool = False
    history_k: int = 0
    history_on: str = "both"
    lag_steps: int = 0
    add_input_derivatives: bool = False
    deriv_order: int = 1
    dt: float = 0.1

    # split
    split_mode: str = "stratified"  # stratified or random
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    stratify_key: str = "hub_u_mean"
    stratify_bins: int = 10

    # model
    hidden_dims: Tuple[int, ...] = (256, 256)
    dropout: float = 0.0
    layer_norm: bool = False

    # train
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 4096
    grad_clip: float = 0.0

    # misc
    verbose: bool = True


def _make_task_config(output_names: List[str], task_groups: Optional[Dict[str, List[str]]] = None) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    groups = task_groups or DEFAULT_TASK_GROUPS
    # keep only present outputs
    groups = {k: [x for x in v if x in output_names] for k, v in groups.items()}
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    task_config = {k: len(v) for k, v in groups.items()}
    return task_config, groups


def _stack_outputs_by_groups(Y: np.ndarray, out_names: List[str], task_groups: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    name2i = {n: i for i, n in enumerate(out_names)}
    out: Dict[str, np.ndarray] = {}
    for k, chs in task_groups.items():
        idx = [name2i[n] for n in chs]
        out[k] = Y[:, idx]
    return out


def _dict_to_matrix(d: Dict[str, torch.Tensor], task_groups: Dict[str, List[str]], device: torch.device) -> torch.Tensor:
    parts = []
    for k, chs in task_groups.items():
        parts.append(d[k].to(device))
    return torch.cat(parts, dim=1)


def compute_scalers(files: List[str], case_cfg: CaseProcessConfig, otf: Optional[OTFFeatureExtractor]) -> Dict[str, np.ndarray]:
    wind_rms = None
    inp_rms = None
    out_rms = None

    for fp in tqdm(files, desc="Scaler stats", ncols=100):
        w, x, y, *_ = process_npz_case(fp, case_cfg, otf=otf)
        if y.shape[0] == 0:
            continue
        if wind_rms is None:
            wind_rms = RunningMeanStd(w.shape[1])
            inp_rms = RunningMeanStd(x.shape[1])
            out_rms = RunningMeanStd(y.shape[1])
        wind_rms.update(w); inp_rms.update(x); out_rms.update(y)

    if wind_rms is None:
        raise ValueError("No effective samples found in provided files. Check history_k/lag_steps and your dataset length.")

    w_mu, w_std = wind_rms.finalize()
    x_mu, x_std = inp_rms.finalize()
    y_mu, y_std = out_rms.finalize()

    return {"wind_mean": w_mu, "wind_std": w_std, "inp_mean": x_mu, "inp_std": x_std, "out_mean": y_mu, "out_std": y_std}


def _norm(a: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (a - mu) / std


@torch.no_grad()
def eval_files(
    model: nn.Module,
    files: List[str],
    case_cfg: CaseProcessConfig,
    task_groups: Dict[str, List[str]],
    scalers: Dict[str, np.ndarray],
    otf: Optional[OTFFeatureExtractor],
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    y_true_list = []
    y_pred_list = []
    out_names = list(case_cfg.desired_output_params)

    for fp in tqdm(files, desc="Eval", ncols=100):
        w, x, y, *_ = process_npz_case(fp, case_cfg, otf=otf)
        if y.shape[0] == 0:
            continue
        w = _norm(w, scalers["wind_mean"], scalers["wind_std"])
        x = _norm(x, scalers["inp_mean"], scalers["inp_std"])
        y_n = _norm(y, scalers["out_mean"], scalers["out_std"])

        n = y.shape[0]
        for s in range(0, n, batch_size):
            ww = torch.from_numpy(w[s:s+batch_size]).to(device)
            xx = torch.from_numpy(x[s:s+batch_size]).to(device)
            pred_dict = model(ww, xx)
            pred = _dict_to_matrix(pred_dict, task_groups, device).cpu().numpy()
            y_pred_list.append(pred)
        y_true_list.append(y_n.astype(np.float32))

    if not y_true_list or not y_pred_list:
        raise ValueError(
            "No valid samples were produced during evaluation. "
            "Common causes: (1) files list is empty; (2) history_k/lag_steps remove all samples; "
            "(3) NPZ wind/features missing and OTF extractor not configured."
        )

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # unnormalize for reporting
    y_true_u = y_true * scalers["out_std"] + scalers["out_mean"]
    y_pred_u = y_pred * scalers["out_std"] + scalers["out_mean"]

    r2_all = r2_score(y_true_u, y_pred_u)
    r2_each = r2_per_output(y_true_u, y_pred_u, out_names)
    return {"r2": float(r2_all)}, {k: float(v) for k, v in r2_each.items()}


def train(cfg: TrainConfig, task_groups: Optional[Dict[str, List[str]]] = None) -> Dict:
    os.makedirs(cfg.output_dir, exist_ok=True)

    all_files = [os.path.join(cfg.data_dir, f) for f in os.listdir(cfg.data_dir) if f.lower().endswith('.npz')]
    all_files.sort()
    if not all_files:
        raise FileNotFoundError(f"No NPZ found in {cfg.data_dir}")

    otf = None
    if cfg.weights_dir is not None:
        otf = OTFFeatureExtractor(OTFFeatureConfig(weights_dir=cfg.weights_dir, device=cfg.device, batch=2048, bts_root=cfg.bts_root))

    if cfg.split_mode == "stratified":
        train_files, val_files, test_files, split_meta = stratified_split_files(
            all_files, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio,
            seed=cfg.split_seed, stratify_key=cfg.stratify_key, stratify_bins=cfg.stratify_bins, otf=otf
        )
    else:
        train_files, val_files, test_files, split_meta = simple_ratio_split(
            all_files, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, seed=cfg.split_seed
        )

    # fix empty/zero-effective splits
    train_files, val_files, test_files, fix_meta = fix_empty_splits(
        train_files, val_files, test_files, history_k=cfg.history_k, lag_steps=cfg.lag_steps, min_effective_samples=1
    )

    # case processing config
    case_cfg = CaseProcessConfig(
        desired_input_params=list(cfg.desired_input_params),
        desired_output_params=list(cfg.desired_output_params),
        angle_mode=cfg.angle_mode,
        keep_raw_angles=cfg.keep_raw_angles,
        history_k=cfg.history_k,
        history_on=cfg.history_on,
        use_all_features=cfg.use_all_features,
        kept_feature_indices=cfg.kept_feature_indices,
        add_input_derivatives=cfg.add_input_derivatives,
        deriv_order=cfg.deriv_order,
        dt=cfg.dt,
        lag_steps=cfg.lag_steps,
    )

    # determine dims from first effective file
    wind0, inp0, out0, wind_names, inp_names, out_names = None, None, None, None, None, None
    for fp in train_files:
        w, x, y, w_names, x_names, y_names = process_npz_case(fp, case_cfg, otf=otf)
        if y.shape[0] > 0:
            wind0, inp0, out0 = w, x, y
            wind_names, inp_names, out_names = w_names, x_names, y_names
            break
    if wind0 is None:
        raise ValueError("Train split has no effective samples; check your preprocessing config.")

    task_config, groups = _make_task_config(out_names, task_groups=task_groups)

    # scalers
    scalers = compute_scalers(train_files, case_cfg, otf=otf)
    np.savez(os.path.join(cfg.output_dir, "scalers.npz"), **scalers)

    # save split/config
    with open(os.path.join(cfg.output_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump({"split_meta": split_meta, "fix_meta": fix_meta, "train": train_files, "val": val_files, "test": test_files}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    # model
    device = torch.device(cfg.device)
    model = DynamicMLPWindTurbineModel(
        in_wind_dim=int(wind0.shape[1]),
        in_input_dim=int(inp0.shape[1]),
        task_config=task_config,
        hidden_dims=list(cfg.hidden_dims),
        dropout=float(cfg.dropout),
        use_layernorm=bool(cfg.layer_norm),
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.MSELoss()

    best_val = -1e9
    best_path = os.path.join(cfg.output_dir, "model_best.pt")

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        running = []

        for fp in tqdm(train_files, desc=f"Epoch {epoch}/{cfg.epochs}", ncols=100):
            w, x, y, *_ = process_npz_case(fp, case_cfg, otf=otf)
            if y.shape[0] == 0:
                continue
            w = _norm(w, scalers["wind_mean"], scalers["wind_std"])
            x = _norm(x, scalers["inp_mean"], scalers["inp_std"])
            y = _norm(y, scalers["out_mean"], scalers["out_std"])

            n = y.shape[0]
            # mini-batch within file
            order = np.random.permutation(n)
            for s in range(0, n, int(cfg.batch_size)):
                idx = order[s:s+int(cfg.batch_size)]
                ww = torch.from_numpy(w[idx]).to(device)
                xx = torch.from_numpy(x[idx]).to(device)
                yy = torch.from_numpy(y[idx]).to(device)

                pred_dict = model(ww, xx)
                pred = _dict_to_matrix(pred_dict, groups, device)
                loss = loss_fn(pred, yy)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                opt.step()
                running.append(float(loss.item()))

        # eval
        val_overall, val_each = eval_files(model, val_files, case_cfg, groups, scalers, otf, batch_size=int(cfg.batch_size), device=device)
        val_r2 = val_overall["r2"]
        if cfg.verbose:
            print(f"[Epoch {epoch}] train_loss={np.mean(running) if running else float('nan'):.6f}  val_r2={val_r2:.4f}")

        if val_r2 > best_val:
            best_val = val_r2
            torch.save({"model": model.state_dict(), "task_config": task_config, "task_groups": groups, "wind_names": wind_names, "inp_names": inp_names, "out_names": out_names}, best_path)

    # final eval (load best)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_overall, test_each = eval_files(model, test_files, case_cfg, groups, scalers, otf, batch_size=int(cfg.batch_size), device=device)

    results = {
        "best_val_r2": float(best_val),
        "test_overall": test_overall,
        "test_each": test_each,
        "task_groups": groups,
        "task_config": task_config,
        "model_path": best_path,
    }
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
