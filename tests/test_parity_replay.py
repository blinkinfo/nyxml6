import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.parity import replay_training_vs_live_parity


class DummyModel:
    def predict(self, x):
        x = np.asarray(x, dtype=float)
        z = x.mean(axis=1) / 10.0
        return 1.0 / (1.0 + np.exp(-z))


def _make_data(seed: int = 7, n5: int = 520):
    rng = np.random.default_rng(seed)
    ts5 = pd.date_range("2026-01-01", periods=n5, freq="5min", tz="UTC")
    close = 50000 + np.cumsum(rng.normal(0, 20, n5))
    open_ = close + rng.normal(0, 5, n5)
    high = np.maximum(open_, close) + rng.uniform(0, 8, n5)
    low = np.minimum(open_, close) - rng.uniform(0, 8, n5)
    vol = rng.uniform(50, 200, n5)
    df5 = pd.DataFrame({"timestamp": ts5, "open": open_, "high": high, "low": low, "close": close, "volume": vol})

    s = df5.set_index("timestamp")
    df15 = s.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    df1h = s.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()

    funding_ts = pd.date_range(ts5.min() - pd.Timedelta("24h"), ts5.max() + pd.Timedelta("4h"), freq="8h", tz="UTC")
    funding = pd.DataFrame({"timestamp": funding_ts, "funding_rate": rng.normal(0, 0.0001, len(funding_ts))})

    cvd = pd.DataFrame({
        "timestamp": ts5,
        "long_taker_size": rng.uniform(100, 500, len(ts5)),
        "short_taker_size": rng.uniform(100, 500, len(ts5)),
        "open_interest": rng.uniform(50000, 60000, len(ts5)),
    })
    return df5, df15, df1h, funding, cvd


def test_replay_training_vs_live_parity_matches_on_synthetic_data():
    df5, df15, df1h, funding, cvd = _make_data()
    model = DummyModel()

    result, details = replay_training_vs_live_parity(
        df5=df5,
        df15=df15,
        df1h=df1h,
        funding=funding,
        cvd=cvd,
        model=model,
        n_checks=60,
        feature_tol=1e-10,
        prob_tol=1e-12,
    )

    assert result.checked_rows == 60
    assert result.feature_mismatch_rows == 0
    assert result.prob_mismatch_rows == 0
    assert result.max_feature_abs_diff <= 1e-10
    assert result.max_prob_abs_diff <= 1e-12
    assert (details["status"] == "ok").all()
