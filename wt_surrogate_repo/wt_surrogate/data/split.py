from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os
import numpy as np

from wt_surrogate.features.otf import OTFFeatureExtractor


def list_npz_files(data_dir: str) -> List[str]:
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.npz')]
    files.sort()
    return files


def compute_stratify_value(fp: str, key: str, otf: Optional[OTFFeatureExtractor] = None) -> float:
    d = np.load(fp, allow_pickle=True)
    if 'features' in d.files:
        feats = d['features']
        names = [str(x) for x in d['feature_names'].tolist()] if 'feature_names' in d.files else [f'feat_{i}' for i in range(feats.shape[1])]
        if key not in names:
            raise KeyError(f"{os.path.basename(fp)}: feature_names missing '{key}'")
        idx = names.index(key)
        return float(np.mean(feats[:, idx]))

    if otf is None:
        raise ValueError(f"{os.path.basename(fp)} has no precomputed features; provide OTFFeatureExtractor for stratify_key='{key}'.")

    n = int(d['output_params'].shape[0]) if 'output_params' in d.files else (int(d['time'].shape[0]) if 'time' in d.files else 0)
    if n <= 0:
        return 0.0
    m = min(n, 2000)
    row_idx = np.linspace(0, n - 1, m, dtype=np.int64)

    feats, names = otf.extract(d, row_idx=row_idx)

    # Backward-compatible aliases for common naming differences across extractor versions
    alias = {
        "hub_mean_u": "hub_u_mean",
        "hub_mean_U": "hub_u_mean",
        "hub_u": "hub_u_mean",
        "hubU_mean": "hub_u_mean",
        "global_mean_u": "global_u_mean",
        "global_mean_U": "global_u_mean",
    }
    key_resolved = alias.get(key, key)
    if key_resolved not in names:
        raise KeyError(f"{os.path.basename(fp)}: extracted features missing '{key}' (resolved='{key_resolved}')")
    idx = names.index(key_resolved)
    return float(np.mean(feats[:, idx]))


def count_effective_samples_in_file(fp: str, history_k: int, lag_steps: int) -> int:
    d = np.load(fp, allow_pickle=True)
    if 'features' in d.files:
        n = int(d['features'].shape[0])
    elif 'output_params' in d.files:
        n = int(d['output_params'].shape[0])
    elif 'time' in d.files:
        n = int(d['time'].shape[0])
    else:
        n = 0
    return max(0, (n - int(lag_steps)) - int(history_k))


def stratified_split_files(
    files: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_key: str,
    stratify_bins: int,
    otf: Optional[OTFFeatureExtractor] = None,
) -> Tuple[List[str], List[str], List[str], Dict]:
    rng = np.random.default_rng(int(seed))
    vals = np.array([compute_stratify_value(fp, stratify_key, otf=otf) for fp in files], dtype=np.float64)

    order = np.argsort(vals)
    bins = max(2, min(int(stratify_bins), len(files)))
    groups = np.array_split(order, bins)

    train: List[str] = []
    val: List[str] = []
    test: List[str] = []
    manifest = {"mode": "stratified_files", "key": stratify_key, "bins": bins, "seed": int(seed), "groups": []}

    for g in groups:
        g = g.tolist()
        rng.shuffle(g)
        n = len(g)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train += [files[i] for i in g[:n_train]]
        val   += [files[i] for i in g[n_train:n_train+n_val]]
        test  += [files[i] for i in g[n_train+n_val:]]

        manifest["groups"].append({
            "n": n,
            "train": n_train,
            "val": n_val,
            "test": n - n_train - n_val,
            "range": [float(vals[g[0]]), float(vals[g[-1]])] if n else None,
        })

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    manifest["counts"] = {"train": len(train), "val": len(val), "test": len(test)}
    return train, val, test, manifest


def simple_ratio_split(files: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str], Dict]:
    rng = np.random.default_rng(int(seed))
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = files[:n_train]
    val = files[n_train:n_train+n_val]
    test = files[n_train+n_val:]
    return train, val, test, {"mode": "random_files", "seed": int(seed), "counts": {"train": len(train), "val": len(val), "test": len(test)}}


def fix_empty_splits(
    train: List[str],
    val: List[str],
    test: List[str],
    history_k: int,
    lag_steps: int,
    min_effective_samples: int = 1,
) -> Tuple[List[str], List[str], List[str], Dict]:
    train = list(train); val = list(val); test = list(test)

    def move_best(src: List[str], dst: List[str]) -> bool:
        if not src:
            return False
        scores = [count_effective_samples_in_file(fp, history_k, lag_steps) for fp in src]
        best_i = int(np.argmax(scores))
        dst.append(src.pop(best_i))
        return True

    def eff(lst: List[str]) -> int:
        return int(sum(count_effective_samples_in_file(fp, history_k, lag_steps) for fp in lst))

    moves = {"to_val": 0, "to_test": 0}

    if not val and len(train) > 1:
        if move_best(train, val):
            moves["to_val"] += 1
    if not test and len(train) > 1:
        if move_best(train, test):
            moves["to_test"] += 1

    while eff(val) < min_effective_samples and len(train) > 1:
        if not move_best(train, val):
            break
        moves["to_val"] += 1
    while eff(test) < min_effective_samples and len(train) > 1:
        if not move_best(train, test):
            break
        moves["to_test"] += 1

    info = {
        "moves": moves,
        "effective": {"train": eff(train), "val": eff(val), "test": eff(test)},
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
    }
    return train, val, test, info
