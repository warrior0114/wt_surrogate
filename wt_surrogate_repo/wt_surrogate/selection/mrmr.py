"""Mutual-information based mRMR feature selection."""
# MIMR_v3_name_based_fixed.py
# mRMR(MI-based) feature selection for your wind-feature dataset (.npz)
# Fixes vs MIMR_v2_name_based.py:
#   1) Fix indentation / NameError (npz_files undefined)
#   2) Prefer feature_names / output_names saved in .npz (no "truth reference" needed except as fallback)
#   3) Memory-safe loading: sample per-file then merge (avoids vstack of multi-million rows)
#   4) Extra sanity checks: consistent names across files, feature dim match

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from joblib import Parallel, delayed

# --- Optional: on-the-fly feature extraction when .npz has no precomputed 'features' ---
import torch
import torch.nn.functional as F
try:
    from wt_surrogate.features.PIFENet import PIFENet
except Exception:
    PIFENet = None
try:
    from wt_surrogate.io.turbsim_bts import read_bts_official
except Exception:
    read_bts_official = None

_EXTRACTOR = None
_EXTRACTOR_DEVICE = None
_EXTRACTOR_WEIGHTS_DIR = None


def _npz_scalar_to_str(x) -> str:
    """Robustly convert npz scalar/0-d array/bytes to python str."""
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            if x.shape == ():
                x = x.item()
            elif x.size == 1:
                x = x.reshape(-1)[0]
                try:
                    x = x.item()
                except Exception:
                    pass
    except Exception:
        pass
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return x.decode(errors="ignore")
    return str(x)


def _get_extractor(weights_dir: str, device: str = 'cpu'):
    """Lazily build PIFENet."""
    global _EXTRACTOR, _EXTRACTOR_DEVICE, _EXTRACTOR_WEIGHTS_DIR
    if _EXTRACTOR is not None and _EXTRACTOR_DEVICE == device and _EXTRACTOR_WEIGHTS_DIR == weights_dir:
        return _EXTRACTOR
   
    if not weights_dir or (not os.path.exists(weights_dir)):
        raise FileNotFoundError(f"weights_dir : {weights_dir}")
    dev = torch.device(device)
    ex = PIFENet(weights_dir=weights_dir).to(dev)
    ex.eval()
    _EXTRACTOR = ex
    _EXTRACTOR_DEVICE = device
    _EXTRACTOR_WEIGHTS_DIR = weights_dir
    return _EXTRACTOR

def _extract_features_from_npz(data: np.lib.npyio.NpzFile, row_idx: np.ndarray, weights_dir: str, device: str = 'cpu') -> Tuple[np.ndarray, List[str]]:
    """Return (features[row_idx], feature_names). Requires wind snapshots or bts_path in npz."""
    ex = _get_extractor(weights_dir, device=device)
    # 1) get wind snapshots
    if 'wind' in data.files and data['wind'] is not None:
        wind_all = data['wind']
        if wind_all.ndim != 4:
            raise ValueError(f"npz wind expects 4D (T,3,ny,nz), got {wind_all.shape}")
        wind = wind_all[row_idx].astype(np.float32, copy=False)
    else:
        if read_bts_official is None:
            raise ImportError("npz has no wind, and cannot import read_bts_official to read .bts")
        if 'bts_path' not in data.files:
            raise KeyError("npz missing wind and bts_path, cannot extract features")
        bts_path = _npz_scalar_to_str(data['bts_path'])
        # If path is relative, resolve against the folder containing the .npz (common in dataset export)
        if not os.path.isabs(bts_path):
            base_dir = _npz_scalar_to_str(data['npz_dir']) if 'npz_dir' in data.files else None
            if base_dir and os.path.isdir(base_dir):
                bts_path = os.path.join(base_dir, bts_path)
        header, wind_data = read_bts_official(bts_path)
        if wind_data is None:
            raise ValueError(f"Failed to read bts: {bts_path}")
        if 'wind_idx' in data.files and data['wind_idx'] is not None:
            idx_all = data['wind_idx'].astype(np.int64)
        else:
            if 'time' not in data.files or 'bts_dt' not in data.files:
                raise KeyError("npz missing wind_idx, and missing time/bts_dt to align BTS")
            t = data['time'].astype(np.float64)
            bts_dt = float(data['bts_dt'])
            idx_all = np.clip(np.round(t / bts_dt).astype(np.int64), 0, wind_data.shape[0] - 1)
        wind = wind_data[idx_all[row_idx]].astype(np.float32, copy=False)

    # 2) interpolate to extractor grid if needed
    w = torch.from_numpy(wind).to(torch.device(device))  # (B,3,ny,nz)
    with torch.no_grad():
        try:
            target_ny = int(ex.hub.shape[-2])
            target_nz = int(ex.hub.shape[-1])
        except Exception:
            target_ny, target_nz = w.shape[-2], w.shape[-1]
        if (w.shape[-2] != target_ny) or (w.shape[-1] != target_nz):
            w = F.interpolate(w, size=(target_ny, target_nz), mode='bilinear', align_corners=False)
        feats = ex(w).detach().cpu().numpy().astype(np.float32, copy=False)

    feat_names = getattr(ex, 'feature_names', [f'f{i}' for i in range(feats.shape[1])])
    feat_names = [str(x) for x in list(feat_names)]
    return feats, feat_names

# ----------------------------------------------------------------------
# Legacy fallback only: used iff .npz does NOT contain output_names
# ----------------------------------------------------------------------
ALL_OUTPUT_COLS_FALLBACK = [
    "TTDspFA", "TTDspSS", "TTDspTwst", "RotTorq", "LSSGagMya", "LSSGagMza",
    "YawBrFxp", "YawBrFyp", "YawBrFzp", "YawBrMxp", "YawBrMyp", "YawBrMzp",
    "TwrBsFxt", "TwrBsFyt", "TwrBsFzt", "TwrBsMxt", "TwrBsMyt", "TwrBsMzt",
    "RootFxb1", "RootFyb1", "RootFzb1", "RootMxb1", "RootMyb1", "RootMzb1",
    "TipDxb1", "TipDyb1", "TipDzb1", "GenPwr", "GenTq"
]


def _as_str_list(arr) -> List[str]:
    if arr is None:
        return []
    return [str(x) for x in list(arr)]


def _list_npz_files(features_dir: str) -> List[str]:
    npz_files = [f for f in os.listdir(features_dir) if f.lower().endswith(".npz")]
    npz_files.sort()
    return npz_files


def _calculate_mi_for_one_column(
    i: int,
    features_discrete: np.ndarray,
    n_features: int,
    random_state: int
) -> Tuple[int, np.ndarray]:
    """Compute MI for upper triangle of row i in the redundancy MI matrix."""
    mi_row = np.zeros(n_features, dtype=np.float64)
    for j in range(i, n_features):
        if i == j:
            mi = 0.0
        else:
            # MI between discrete feature Xi and discrete feature Xj (treat Xj as "class label")
            mi = mutual_info_classif(
                features_discrete[:, i].reshape(-1, 1),
                features_discrete[:, j],
                discrete_features=True,
                random_state=random_state
            )[0]
        mi_row[j] = mi
    return i, mi_row


def calculate_feature_mi_matrix_parallel(
    df_features: pd.DataFrame,
    n_bins: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
    backend: str = "threading",
) -> pd.DataFrame:
    """
    Parallel redundancy MI matrix among features.
    backend:
      - 'threading': lower overhead, no big data copy
      - 'loky': safer RNG isolation, but may copy data to processes
    """
    print("Calculating feature mutual information matrix (redundancy) in parallel...")
    n_features = df_features.shape[1]

    # Discretize
    try:
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="uniform",
            subsample=200_000,
            random_state=random_state
        )
    except TypeError:
        # Older sklearn may not have 'subsample'
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="uniform"
        )

    features_discrete = discretizer.fit_transform(df_features).astype(np.int32, copy=False)

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_calculate_mi_for_one_column)(i, features_discrete, n_features, random_state)
        for i in tqdm(range(n_features), desc="Dispatching MI tasks")
    )

    mi_matrix = np.zeros((n_features, n_features), dtype=np.float64)
    for i, mi_row_upper in results:
        for j in range(i, n_features):
            v = float(mi_row_upper[j])
            mi_matrix[i, j] = v
            mi_matrix[j, i] = v

    return pd.DataFrame(mi_matrix, index=df_features.columns, columns=df_features.columns)


def _sample_rows_from_one_file(
    fp: str,
    target_indices: List[int],
    max_rows: int,
    rng: np.random.Generator,
    ref_output_names: Optional[List[str]] = None,
    ref_feature_names: Optional[List[str]] = None,
    weights_dir: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load one .npz and sample up to max_rows rows. Returns (X, Y, feature_names, output_names).

    Supports two formats:
      1) Old format: .npz contains precomputed 'features' / 'feature_names'
      2) New format: No 'features', contains 'wind' (or bts_path + wind_idx/time), features extracted on demand.
    """
    data = np.load(fp, allow_pickle=True)

    if "output_params" not in data.files:
        raise KeyError(f"{os.path.basename(fp)} missing output_params")

    Yfull = data["output_params"]
    output_names = _as_str_list(data["output_names"]) if "output_names" in data.files else _as_str_list(ALL_OUTPUT_COLS_FALLBACK)

    if Yfull.ndim != 2:
        raise ValueError(f"{os.path.basename(fp)}: output_params expects 2D array, got {Yfull.shape}")

    Y = Yfull[:, target_indices]
    n = int(Y.shape[0])
    if n <= 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, len(target_indices)), dtype=np.float32),
            [],
            output_names,
        )

    if n <= max_rows:
        row_idx = np.arange(n, dtype=np.int64)
    else:
        row_idx = rng.choice(n, size=max_rows, replace=False).astype(np.int64)

    if "features" in data.files:
        Xfull = data["features"]
        if Xfull.ndim != 2:
            raise ValueError(f"{os.path.basename(fp)}: features expects 2D array, got {Xfull.shape}")
        X = Xfull[row_idx].astype(np.float32, copy=False)
        feature_names = _as_str_list(data["feature_names"]) if "feature_names" in data.files else [f"feat_{i}" for i in range(X.shape[1])]
    else:
        if weights_dir is None:
            raise ValueError("npz does not contain features, weights_dir is required for on-the-fly extraction")
        X, feature_names = _extract_features_from_npz(data, row_idx, weights_dir=weights_dir, device=device)

    if ref_output_names is not None and ref_output_names and output_names != ref_output_names:
        raise ValueError(f"output_names mismatch in {os.path.basename(fp)}")
    if ref_feature_names is not None and ref_feature_names and feature_names != ref_feature_names:
        raise ValueError(f"feature_names mismatch in {os.path.basename(fp)}")

    return X, Y[row_idx].astype(np.float32, copy=False), feature_names, output_names


def select_features_mimr(
    features_dir: str,
    target_cols: List[str],
    beta: float = 0.5,
    num_features_to_select: Optional[int] = None,
    selection_score_threshold: Optional[float] = None,
    subsample_size: int = 50000,
    per_file_cap: Optional[int] = None,
    max_files: Optional[int] = None,
    random_state: int = 42,
    n_bins: int = 10,
    mi_backend: str = "threading",
    weights_dir: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[List[int], pd.DataFrame]:
    """
    mRMR selection.
    - Uses .npz 'feature_names' and 'output_names' if present (recommended)
    - Loads files memory-safely by sampling up to per_file_cap rows from each file,
      then globally downsamples to subsample_size if still too large.
    """
    print(f"--- Starting mRMR feature selection process (beta={beta}) ---")

    npz_files = _list_npz_files(features_dir)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in directory: {features_dir}")

    # Optional limit to speed up selection in large datasets
    if max_files is not None:
        try:
            mf = int(max_files)
            if mf > 0:
                npz_files = npz_files[:mf]
        except Exception:
            pass

    rng = np.random.default_rng(random_state)

    # Read names & build target indices from first file
    first_fp = os.path.join(features_dir, npz_files[0])
    first = np.load(first_fp, allow_pickle=True)
    output_names = _as_str_list(first["output_names"]) if "output_names" in first.files else _as_str_list(ALL_OUTPUT_COLS_FALLBACK)
    feature_names = _as_str_list(first["feature_names"]) if "feature_names" in first.files else None

    out_map: Dict[str, int] = {name: i for i, name in enumerate(output_names)}
    missing = [c for c in target_cols if c not in out_map]
    if missing:
        raise KeyError(f"Target columns missing: {missing}. output_names={output_names}")

    target_indices = [out_map[name] for name in target_cols]

    # Determine per-file cap (balanced across cases)
    if per_file_cap is None:
        # heuristic: 3x the equal-share, at least 500, at most 20000
        per_file_cap = int(np.clip(np.ceil(subsample_size / max(len(npz_files), 1)) * 3, 500, 20000))

    print(f"Sampling strategy: Max {per_file_cap} rows per file; Global cap subsample_size={subsample_size}")

    # Load sampled rows per file
    X_list, Y_list = [], []
    ref_output_names = output_names
    ref_feature_names = feature_names

    for f in tqdm(npz_files, desc="Reading and sampling .npz"):
        fp = os.path.join(features_dir, f)
        Xi, Yi, feat_names_i, out_names_i = _sample_rows_from_one_file(
            fp=fp,
            target_indices=target_indices,
            max_rows=per_file_cap,
            rng=rng,
            ref_output_names=ref_output_names,
            ref_feature_names=ref_feature_names if ref_feature_names is not None else None,
            weights_dir=weights_dir,
            device=device,
        )
        if ref_feature_names is None:
            ref_feature_names = feat_names_i  # adopt from first sampled file
        X_list.append(Xi)
        Y_list.append(Yi)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    # Global downsample if needed
    if X.shape[0] > subsample_size:
        idx = rng.choice(X.shape[0], size=subsample_size, replace=False)
        X = X[idx]
        Y = Y[idx]

    feature_names = ref_feature_names if ref_feature_names is not None else [f"feat_{i}" for i in range(X.shape[1])]

    df_features = pd.DataFrame(X, columns=feature_names)
    df_targets = pd.DataFrame(Y, columns=target_cols)
    print(f"Data size for MI calculation: {df_features.shape[0]} samples, {df_features.shape[1]} features")

    # Step 1: MI(feature; target) relevance
    print("\nStep 1: Calculating Mutual Information (Feature; Target) Relevance...")
    mi_scores = pd.DataFrame(index=df_features.columns)
    for target in tqdm(target_cols, desc="Calculating MI(Feature;Target)"):
        mi_scores[target] = mutual_info_regression(df_features, df_targets[target], random_state=random_state)
    mi_scores["relevance"] = mi_scores.mean(axis=1)

    # Step 2: redundancy matrix
    mi_feature_matrix = calculate_feature_mi_matrix_parallel(
        df_features, n_bins=n_bins, random_state=random_state, backend=mi_backend
    )

    # Step 3: iterative selection
    print("\nStep 3: Iterative mRMR selection...")
    remaining = list(df_features.columns)
    selected: List[str] = []
    selection_info = []

    first_feat = mi_scores["relevance"].idxmax()
    selected.append(first_feat)
    remaining.remove(first_feat)
    selection_info.append({
        "rank": 1,
        "feature": first_feat,
        "selection_score": float(mi_scores.loc[first_feat, "relevance"]),
        "relevance": float(mi_scores.loc[first_feat, "relevance"]),
        "redundancy": 0.0
    })

    if num_features_to_select is not None:
        iterations = max(num_features_to_select - 1, 0)
        print(f"Selecting fixed number of features: {num_features_to_select}")
    else:
        iterations = len(remaining)
        print(f"Using threshold stopping criteria, threshold={selection_score_threshold}")

    for k in tqdm(range(iterations), desc="Selecting best features"):
        if not remaining:
            break

        best = None
        for cand in remaining:
            rel = float(mi_scores.loc[cand, "relevance"])
            red = float(mi_feature_matrix.loc[cand, selected].mean()) if selected else 0.0
            score = rel - beta * red
            if (best is None) or (score > best["score"]):
                best = {"feature": cand, "score": score, "relevance": rel, "redundancy": red}

        assert best is not None

        if selection_score_threshold is not None and num_features_to_select is None:
            if best["score"] < selection_score_threshold:
                print(f"\nStopping: Best candidate '{best['feature']}' score {best['score']:.6f} < threshold {selection_score_threshold}")
                break

        selected.append(best["feature"])
        remaining.remove(best["feature"])
        selection_info.append({
            "rank": len(selected),
            "feature": best["feature"],
            "selection_score": float(best["score"]),
            "relevance": float(best["relevance"]),
            "redundancy": float(best["redundancy"]),
        })

    final_df = pd.DataFrame(selection_info).set_index("rank")

    # Map back to original index
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    final_df["original_index"] = final_df["feature"].map(name_to_idx)

    kept_indices = sorted(final_df["original_index"].tolist())

    print("\n" + "=" * 80)
    print("mRMR Final Results")
    print("=" * 80)
    print(final_df[["feature", "original_index", "selection_score", "relevance", "redundancy"]])
    print("\nRecommended feature indices to keep (sorted):")
    print(kept_indices)

    # Save full score table
    all_scores_df = mi_scores[["relevance"]].copy()
    sel_scores = final_df[["feature", "selection_score", "redundancy"]].set_index("feature")
    all_scores_df = all_scores_df.merge(sel_scores, left_index=True, right_index=True, how="left")
    all_scores_df = all_scores_df.reset_index().rename(columns={"index": "feature_name"})
    all_scores_df["original_index"] = all_scores_df["feature_name"].map(name_to_idx)
    all_scores_df = all_scores_df.sort_values("original_index").reset_index(drop=True)
    all_scores_df = all_scores_df[["original_index", "feature_name", "relevance", "redundancy", "selection_score"]]

    out_csv = os.path.join(features_dir, "mimr_feature_scores.csv")
    all_scores_df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nAll feature scores saved to: {out_csv}")

    return kept_indices, final_df