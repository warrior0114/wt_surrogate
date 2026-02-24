# wt-surrogate

Wind turbine dynamic-response surrogate model + wind-field feature engineering (PIFENet).

This repository is a cleaned-up, open-source friendly refactor of the original research scripts.

## Folder layout

- `wt_surrogate/` : importable Python package
  - `io/` : OpenFAST `.out` and TurbSim `.bts` readers
  - `features/` : PIFENet + preprocessing + on-the-fly feature extraction
  - `data/` : NPZ dataset builder + dataset helpers + splitting utilities
  - `models/` : DynamicMLPWindTurbineModel
  - `selection/` : MI-based mRMR feature selection
  - `training/` : training pipeline + metrics/scalers
- `scripts/` : command-line entrypoints

## Quickstart

### 1) Build NPZ dataset

```bash
python scripts/build_npz_dataset.py \
  --bts_dir /path/to/bts \
  --openfast_out_root /path/to/openfast_out \
  --output_dir /path/to/npz_out \
  --t_start 100 --t_end 700 --dt 0.1
```

### 2) Optional: mRMR feature selection (OTF)

```bash
python scripts/mrmr_select.py \
  --npz_dir /path/to/npz_out \
  --weights_dir /path/to/PIFENet_weights \
  --n_select 60 \
  --out_json kept_features.json
```

### 3) Train surrogate (OTF or precomputed features)

```bash
python scripts/train_surrogate.py \
  --data_dir /path/to/npz_out \
  --output_dir runs/exp01 \
  --weights_dir /path/to/PIFENet_weights \
  --kept_features_json /path/to/npz_out/kept_features.json \
  --epochs 50 --batch_size 4096
```

Outputs:
- `runs/exp01/model_best.pt`
- `runs/exp01/scalers.npz`
- `runs/exp01/split.json`
- `runs/exp01/metrics.json`

## Notes

- If your NPZ files do **not** store `wind`, you must supply `--weights_dir` and (optionally) `--bts_root` so the on-the-fly extractor can locate `.bts`.
- For reproducibility, store your PIFENet weights in a versioned folder and document the grid parameters.

## License

MIT (see `LICENSE`).
