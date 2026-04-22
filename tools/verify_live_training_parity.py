#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml import data_fetcher, model_store
from ml.parity import replay_training_vs_live_parity


def main() -> int:
    p = argparse.ArgumentParser(description="Replay training vs live feature/probability parity")
    p.add_argument("--months", type=int, default=2, help="History window to fetch")
    p.add_argument("--checks", type=int, default=200, help="Rows to compare from tail")
    p.add_argument("--slot", default="current", choices=["current", "candidate"], help="Model slot")
    p.add_argument("--out", default="models/reports/parity_replay.csv", help="CSV output path")
    args = p.parse_args()

    model = model_store.load_model(args.slot)
    if model is None:
        print(f"ERROR: model slot '{args.slot}' not found")
        return 2

    try:
        data = data_fetcher.fetch_all(months=args.months)
    except Exception as exc:
        print(f"ERROR: failed to fetch market data for parity replay: {exc}")
        return 3
    result, df = replay_training_vs_live_parity(
        df5=data["df5"],
        df15=data["df15"],
        df1h=data["df1h"],
        funding=data["funding"],
        cvd=data["cvd"],
        model=model,
        n_checks=args.checks,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("=== PARITY REPLAY SUMMARY ===")
    print(f"checked_rows          : {result.checked_rows}")
    print(f"feature_mismatch_rows : {result.feature_mismatch_rows}")
    print(f"prob_mismatch_rows    : {result.prob_mismatch_rows}")
    print(f"max_feature_abs_diff  : {result.max_feature_abs_diff:.12g}")
    print(f"max_prob_abs_diff     : {result.max_prob_abs_diff:.12g}")
    print(f"report_csv            : {out_path}")

    if result.feature_mismatch_rows or result.prob_mismatch_rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
