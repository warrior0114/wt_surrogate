from __future__ import annotations
import argparse
from wt_surrogate.data.build_dataset import BuildDatasetConfig, build_npz_dataset

def main():
    ap = argparse.ArgumentParser(description="Build per-case NPZ dataset from OpenFAST .out + TurbSim .bts")
    ap.add_argument("--bts_dir", required=True, help="Root directory containing .bts files (recursive search).")
    ap.add_argument("--openfast_out_root", required=True, help="Root directory containing OpenFAST ASCII .out files.")
    ap.add_argument("--output_dir", required=True, help="Output directory for NPZ + manifest.csv")
    ap.add_argument("--t_start", type=float, default=100.0)
    ap.add_argument("--t_end", type=float, default=700.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--store_wind", action="store_true", help="Save wind snapshots into NPZ (default True).")
    ap.add_argument("--no_store_wind", action="store_true", help="Do not save wind snapshots (compute OTF from BTS later).")
    ap.add_argument("--wind_dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    cfg = BuildDatasetConfig(
        bts_dir=args.bts_dir,
        openfast_out_root=args.openfast_out_root,
        output_dir=args.output_dir,
        time_start=args.t_start,
        time_end=args.t_end,
        time_dt=args.dt,
        store_wind=(False if args.no_store_wind else True) if (args.store_wind or args.no_store_wind) else True,
        wind_save_dtype=args.wind_dtype,
    )
    manifest = build_npz_dataset(cfg)
    print(f"Saved manifest: {manifest}")

if __name__ == "__main__":
    main()
