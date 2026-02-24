from __future__ import annotations
import argparse
import json
import os

from wt_surrogate.training.pipeline import TrainConfig, train


def main():
    ap = argparse.ArgumentParser(description="Train dynamic surrogate model from NPZ dataset.")
    ap.add_argument("--data_dir", required=True, help="Directory containing NPZ files.")
    ap.add_argument("--output_dir", required=True, help="Directory to store checkpoints/metrics.")
    ap.add_argument("--weights_dir", default=None, help="PIFENet weights dir (required if NPZ has no 'features').")
    ap.add_argument("--bts_root", default=None, help="Root for resolving relative BTS paths when NPZ doesn't store wind.")
    ap.add_argument("--kept_features_json", default=None, help="JSON from mrmr_select.py containing kept_feature_indices")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--history_k", type=int, default=0)
    ap.add_argument("--lag_steps", type=int, default=0)
    ap.add_argument("--split_mode", choices=["stratified","random"], default="stratified")
    ap.add_argument("--stratify_key", default="hub_u_mean")
    ap.add_argument("--stratify_bins", type=int, default=10)
    args = ap.parse_args()

    kept_indices = None
    if args.kept_features_json:
        with open(args.kept_features_json, "r", encoding="utf-8") as f:
            kept_indices = json.load(f).get("kept_feature_indices")

    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        bts_root=args.bts_root,
        kept_feature_indices=kept_indices,
        device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        history_k=args.history_k,
        lag_steps=args.lag_steps,
        split_mode=args.split_mode,
        stratify_key=args.stratify_key,
        stratify_bins=args.stratify_bins,
    )

    results = train(cfg)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
