from __future__ import annotations
import argparse
import json
import os
from wt_surrogate.selection.mrmr import select_features_mimr

def main():
    ap = argparse.ArgumentParser(description="mRMR (MI-based) feature selection on NPZ dataset.")
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--weights_dir", required=True, help="PIFENet weights directory (for on-the-fly extraction).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--desired_output", nargs="+", default=["RootFxb1","RootMyb1","TwrBsFxt","TwrBsMyt","TTDspFA","GenPwr"])
    ap.add_argument("--n_select", type=int, default=60)
    ap.add_argument("--max_files", type=int, default=50)
    ap.add_argument("--sample_per_file", type=int, default=3000)
    ap.add_argument("--out_json", default="kept_features.json")
    args = ap.parse_args()

    kept_indices, kept_names, info = select_features_mimr(
        npz_dir=args.npz_dir,
        weights_dir=args.weights_dir,
        desired_output_params=args.desired_output,
        n_select=args.n_select,
        max_files=args.max_files,
        sample_per_file=args.sample_per_file,
        device=args.device,
    )

    out_path = args.out_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(args.npz_dir, out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"kept_feature_indices": kept_indices, "kept_feature_names": kept_names, "info": info}, f, indent=2, ensure_ascii=False)

    print(f"Saved kept indices to: {out_path}")
    print(f"Selected {len(kept_indices)} features.")

if __name__ == "__main__":
    main()
