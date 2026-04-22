"""Training-vs-live replay parity checks for ML feature/probability paths."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLS, build_features, build_live_features


@dataclass
class ParityResult:
    checked_rows: int
    feature_mismatch_rows: int
    prob_mismatch_rows: int
    max_feature_abs_diff: float
    max_prob_abs_diff: float


def _row_hash(row: np.ndarray) -> str:
    return sha256(np.asarray(row, dtype=np.float64).tobytes()).hexdigest()[:16]


def replay_training_vs_live_parity(
    *,
    df5: pd.DataFrame,
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    funding: pd.DataFrame,
    cvd: pd.DataFrame | None,
    model,
    n_checks: int = 200,
    feature_tol: float = 1e-12,
    prob_tol: float = 1e-12,
) -> tuple[ParityResult, pd.DataFrame]:
    """Replay historical rows through build_live_features and compare to training.

    For each selected training feature row at timestamp t, reconstruct a live-like
    snapshot that ends at candle open t (the still-forming current 5m bar in
    production), run build_live_features(), and compare both feature vector and
    model probability.
    """
    feat_df = build_features(df5, df15, df1h, funding, cvd)
    if feat_df.empty:
        raise ValueError("No feature rows available for parity replay")

    checks = feat_df.tail(max(1, int(n_checks))).reset_index(drop=True)

    rows: list[dict] = []
    max_feature_abs_diff = 0.0
    max_prob_abs_diff = 0.0
    feature_mismatch_rows = 0
    prob_mismatch_rows = 0

    for row in checks.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        # Align to production semantics:
        # MLStrategy fetches [start_ms, slot_end_ms) where slot_end_ms is the
        # NEXT slot open. Therefore the latest 5m row included is the current
        # still-forming candle at timestamp t, not t+5m.
        live_end = ts

        df5_live = df5[df5["timestamp"] <= live_end].copy()
        df15_live = df15[df15["timestamp"] <= live_end].copy()
        df1h_live = df1h[df1h["timestamp"] <= live_end].copy()
        cvd_live = cvd[cvd["timestamp"] <= live_end].copy() if cvd is not None and not cvd.empty else None

        funding_hist = funding[funding["timestamp"] <= ts][["timestamp", "funding_rate"]].copy()
        funding_hist = funding_hist.sort_values("timestamp").tail(24)
        funding_records = deque(
            [
                {"timestamp": pd.Timestamp(r.timestamp), "funding_rate": float(r.funding_rate)}
                for r in funding_hist.itertuples(index=False)
            ],
            maxlen=24,
        )
        funding_rate_float = float(funding_hist["funding_rate"].iloc[-1]) if not funding_hist.empty else None

        live_row, nan_feats = build_live_features(
            df5_live=df5_live,
            df15_live=df15_live,
            df1h_live=df1h_live,
            funding_rate_float=funding_rate_float,
            funding_records=funding_records,
            cvd_live=cvd_live,
        )
        if live_row is None:
            rows.append({
                "timestamp": ts,
                "status": "live_none",
                "nan_features": ",".join(nan_feats),
            })
            feature_mismatch_rows += 1
            prob_mismatch_rows += 1
            continue

        train_row = np.asarray([getattr(row, c) for c in FEATURE_COLS], dtype=np.float64).reshape(1, -1)
        feature_abs_diff = float(np.max(np.abs(train_row - live_row)))
        max_feature_abs_diff = max(max_feature_abs_diff, feature_abs_diff)

        train_prob = float(model.predict(train_row)[0])
        live_prob = float(model.predict(live_row)[0])
        prob_abs_diff = abs(train_prob - live_prob)
        max_prob_abs_diff = max(max_prob_abs_diff, prob_abs_diff)

        feat_ok = feature_abs_diff <= feature_tol
        prob_ok = prob_abs_diff <= prob_tol
        if not feat_ok:
            feature_mismatch_rows += 1
        if not prob_ok:
            prob_mismatch_rows += 1

        rows.append({
            "timestamp": ts,
            "status": "ok" if (feat_ok and prob_ok) else "mismatch",
            "feature_abs_diff": feature_abs_diff,
            "prob_abs_diff": prob_abs_diff,
            "train_prob": train_prob,
            "live_prob": live_prob,
            "train_hash": _row_hash(train_row[0]),
            "live_hash": _row_hash(live_row[0]),
        })

    result = ParityResult(
        checked_rows=len(rows),
        feature_mismatch_rows=feature_mismatch_rows,
        prob_mismatch_rows=prob_mismatch_rows,
        max_feature_abs_diff=max_feature_abs_diff,
        max_prob_abs_diff=max_prob_abs_diff,
    )
    return result, pd.DataFrame(rows)
