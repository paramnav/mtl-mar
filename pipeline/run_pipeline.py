#!/usr/bin/env python3
"""
Master pipeline runner for mtl-mar.

Run feature building → training → evaluation → maps with a single command.
Nothing in the existing codebase is modified; all outputs go to pipeline/output/.

Examples:
    python pipeline/run_pipeline.py
    python pipeline/run_pipeline.py --use-existing-features --skip-hp-sweep --skip-maps
    python pipeline/run_pipeline.py --target MAR --models rf xgb --use-existing-features
    python pipeline/run_pipeline.py --use-existing-features --dnn --dnn-arch attention
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))


def main():
    parser = argparse.ArgumentParser(
        description="mtl-mar: end-to-end pipeline (new data → results)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--target", type=str, default=None, choices=["MAR", "TOC"],
        help="Process only MAR or TOC (default: both)",
    )
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-tree", action="store_true")
    parser.add_argument("--skip-dnn", action="store_true", dest="skip_dnn")
    parser.add_argument(
        "--dnn", action="store_true",
        help="Include DNN step (off by default)",
    )
    parser.add_argument(
        "--use-existing-features", action="store_true",
        help="Use features from data/interim/ (skip step 1)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Build features for a single dataset only (e.g. MAR_CS)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["rf", "xgb", "cb", "knn"],
        help="Tree models: rf, xgb, cb, knn",
    )
    parser.add_argument("--skip-hp-sweep", action="store_true")
    parser.add_argument("--skip-maps", action="store_true")
    parser.add_argument(
        "--dnn-arch", type=str, default="mlp", choices=["mlp", "attention"],
    )
    parser.add_argument(
        "--mc-samples", type=int, default=100,
        help="MC-dropout samples for DNN uncertainty (0=off)",
    )

    args = parser.parse_args()
    t0 = time.time()

    if not args.skip_features and not args.use_existing_features:
        print("\n" + "█" * 80)
        print("  STEP 1: Building training features from raw data")
        print("█" * 80 + "\n")
        from step1_build_features import build_dataset
        import config as cfg

        if args.dataset:
            build_dataset(args.dataset)
        elif args.target:
            for key in cfg.LABEL_FILES:
                if key.startswith(args.target):
                    build_dataset(key)
        else:
            for key in cfg.LABEL_FILES:
                build_dataset(key)
    else:
        if args.use_existing_features:
            print("\n→ Using existing features from data/interim/ (skipping step 1)")
        else:
            print("\n→ Skipping step 1 (feature building)")

    if not args.skip_tree:
        print("\n" + "█" * 80)
        print("  STEP 2: Tree-based models (train + evaluate + maps)")
        print("█" * 80 + "\n")
        from step2_train_and_evaluate import run_for_target

        targets = [args.target] if args.target else ["MAR", "TOC"]
        for t in targets:
            run_for_target(
                t, args.models, args.skip_hp_sweep, args.skip_maps, args.use_existing_features
            )
    else:
        print("\n→ Skipping step 2 (tree-based models)")

    if args.dnn and not args.skip_dnn:
        print("\n" + "█" * 80)
        print("  STEP 3: DNN models (train + evaluate + maps)")
        print("█" * 80 + "\n")
        from step3_train_dnn import run_dnn_for_target

        targets = [args.target] if args.target else ["MAR", "TOC"]
        for t in targets:
            run_dnn_for_target(
                t, args.dnn_arch, args.skip_maps, args.use_existing_features, args.mc_samples
            )
    elif not args.dnn:
        print("\n→ Skipping step 3 (DNN) — pass --dnn to include")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Outputs: {ROOT / 'pipeline' / 'output'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
