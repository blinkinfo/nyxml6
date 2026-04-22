"""Feature engineering for LightGBM ML strategy — BLUEPRINT sections 4, 5, 6.

ZERO lookahead bias: all features use shift(k>=1). Target uses shift(-1) (future,
only for training labels — never used as a feature).

Target semantics: 1 if the NEXT candle closes at or above its own open
(close[i+1] >= open[i+1]), matching Polymarket's settlement logic
(resolver.py: winner = "Up" if close_price >= open_price else "Down").

42 features total: candle shape (7), volume (2), 15m context (3), 1h context (3),
funding (2), OHLCV pressure (5), time-of-day cyclical (4), volatility regime (2),
momentum (4: rsi14, candle_streak, price_in_range, ema_cross_5m),
structure (3: body_vs_range5, range_expansion, vwap_dist_20),
Gate.io CVD taker flow (7: cvd_ratio, cvd_delta_norm, cvd_cumulative_5,
cvd_cumulative_20, cvd_trend_slope, cvd_divergence, oi_change_5bar).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column order — MUST match exactly (42 features)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "body_ratio_n1", "body_ratio_n2", "body_ratio_n3",
    "upper_wick_n1", "upper_wick_n2",
    "lower_wick_n1", "lower_wick_n2",
    "volume_ratio_n1", "volume_ratio_n2",
    "body_ratio_15m", "dir_15m", "volume_ratio_15m",
    "body_ratio_1h", "dir_1h", "ema9_slope_1h",
    "funding_rate", "funding_zscore",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "vol_zscore", "vol_trend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",  # cyclical time (replaces hour_utc, dow)
    "atr_percentile_24h", "vol_regime",
    "rsi14", "candle_streak", "price_in_range", "ema_cross_5m",  # momentum features
    # structure features
    "body_vs_range5", "range_expansion", "vwap_dist_20",
    # Gate.io CVD taker flow features (indices 35-41)
    "cvd_ratio", "cvd_delta_norm",
    # CVD accumulation + open interest features (indices 37-41)
    "cvd_cumulative_5", "cvd_cumulative_20", "cvd_trend_slope",
    "cvd_divergence", "oi_change_5bar",
]


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    """ATR14 using EWM (BLUEPRINT spec)."""
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()


def _asof_backward(left_ts: pd.Series, right: pd.DataFrame, right_cols: list[str]) -> pd.DataFrame:
    """
    Backward-fill lookup: for each timestamp in left_ts, find the last row in
    right where right['timestamp'] <= left_ts.  Uses pd.merge_asof (vectorized,
    C-level) instead of a Python row loop — identical semantics, ~100x faster.

    left_ts  : Series of tz-aware timestamps (may contain NaT), any name.
    right    : sorted DataFrame with a 'timestamp' column + right_cols.
    Returns  : DataFrame indexed 0..len(left_ts)-1 with right_cols,
               NaN where no prior right row exists or left_ts is NaT.
    """
    n = len(left_ts)

    # Build a left frame with a positional index column so we can reindex after
    # the merge.  Give the key a unique name to avoid collisions with `right`.
    left_df = pd.DataFrame({"_left_ts": left_ts.values, "_pos": np.arange(n)})

    # Ensure both key columns share the exact same dtype before merge_asof.
    # Localize if tz-naive, then cast to datetime64[ms, UTC].
    col = left_df["_left_ts"]
    if col.dt.tz is None:
        col = col.dt.tz_localize("UTC")
    left_df["_left_ts"] = col.astype("datetime64[ms, UTC]")
    right = right.copy()
    ts_col = right["timestamp"]
    if ts_col.dt.tz is None:
        ts_col = ts_col.dt.tz_localize("UTC")
    right["timestamp"] = ts_col.astype("datetime64[ms, UTC]")

    # pd.merge_asof refuses NaT in the left key (raises ValueError).
    # ts_n1 = df5["timestamp"].shift(1) always produces NaT at row 0.
    # Solution: filter those rows out, merge the valid subset, then reindex
    # back to the full 0..n-1 range — NaT positions stay NaN in output.
    valid_mask = left_df["_left_ts"].notna()
    left_valid = left_df[valid_mask].reset_index(drop=True)

    if left_valid.empty:
        # All rows were NaT — return all-NaN frame of correct shape.
        return pd.DataFrame(np.nan, index=np.arange(n), columns=right_cols)

    merged = pd.merge_asof(
        left_valid,
        right[["timestamp"] + right_cols],
        left_on="_left_ts",
        right_on="timestamp",
        direction="backward",
    )

    # Restore original positions: set _pos as index, reindex to 0..n-1.
    # Rows that were NaT (excluded above) will have NaN filled automatically.
    result = (
        merged[["_pos"] + right_cols]
        .set_index("_pos")
        .reindex(np.arange(n))
    )
    return result.reset_index(drop=True)


def build_features(
    df5: pd.DataFrame,
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    funding: pd.DataFrame,
    cvd: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build 42 features per BLUEPRINT sections 4-6. Returns df with FEATURE_COLS + 'target'.

    Args:
        df5:     5m OHLCV candles from MEXC spot.
        df15:    15m OHLCV candles from MEXC futures.
        df1h:    1h OHLCV candles from MEXC futures.
        funding: Funding rate history from MEXC futures.
        cvd:     Gate.io 5m taker volume DataFrame with columns:
                 timestamp, long_taker_size, short_taker_size, open_interest.
                 Used to compute 7 CVD/OI features:
                   - cvd_ratio:         long_taker_size / (long + short), clamped [0, 1]
                   - cvd_delta_norm:    (long - short) / ATR5, ATR-normalised delta
                   - cvd_cumulative_5:  rolling 5-bar sum of delta / ATR5, shift(1)
                   - cvd_cumulative_20: rolling 20-bar sum of delta / ATR5, shift(1)
                   - cvd_trend_slope:   OLS slope of delta over rolling 10-bar window / ATR5, shift(1)
                   - cvd_divergence:    +1 price/CVD disagree, -1 agree, 0 flat (5-bar window, shift(1))
                   - oi_change_5bar:    (oi[t-1] - oi[t-6]) / |oi[t-6]|, pct change over 5 bars
                 If None or empty, cvd_ratio defaults to 0.5, cvd_delta_norm to 0.0,
                 and all 5 accumulation/OI features default to 0.0
                 (neutral — no directional information available).
    """

    # Work on copies with clean RangeIndex
    df5 = df5.copy().reset_index(drop=True)
    df15 = df15.copy().reset_index(drop=True)
    df1h = df1h.copy().reset_index(drop=True)
    funding = funding.copy().reset_index(drop=True)

    # Sort ascending (should already be sorted, but be safe)
    df5 = df5.sort_values("timestamp").reset_index(drop=True)
    df15 = df15.sort_values("timestamp").reset_index(drop=True)
    df1h = df1h.sort_values("timestamp").reset_index(drop=True)
    funding = funding.sort_values("timestamp").reset_index(drop=True)

    # Normalize all timestamps to ms UTC for consistent merging
    for df in [df5, df15, df1h, funding]:
        df["timestamp"] = df["timestamp"].astype("datetime64[ms, UTC]")

    # -----------------------------------------------------------------------
    # 5m features — all use shift(k>=1), NEVER shift(0)
    # -----------------------------------------------------------------------
    atr5 = compute_atr14(df5)

    df5["body_ratio_n1"] = (df5["close"].shift(1) - df5["open"].shift(1)) / atr5.shift(1)
    df5["body_ratio_n2"] = (df5["close"].shift(2) - df5["open"].shift(2)) / atr5.shift(2)
    df5["body_ratio_n3"] = (df5["close"].shift(3) - df5["open"].shift(3)) / atr5.shift(3)

    df5["upper_wick_n1"] = (
        df5["high"].shift(1) - df5[["open", "close"]].shift(1).max(axis=1)
    ) / atr5.shift(1)
    df5["upper_wick_n2"] = (
        df5["high"].shift(2) - df5[["open", "close"]].shift(2).max(axis=1)
    ) / atr5.shift(2)

    df5["lower_wick_n1"] = (
        df5[["open", "close"]].shift(1).min(axis=1) - df5["low"].shift(1)
    ) / atr5.shift(1)
    df5["lower_wick_n2"] = (
        df5[["open", "close"]].shift(2).min(axis=1) - df5["low"].shift(2)
    ) / atr5.shift(2)

    # volume_ratio_n1: N-1 volume divided by rolling mean of the 20 candles
    # ending at N-2 (i.e. vol[i-2]..vol[i-21]).
    # shift(2).rolling(20) at row i = mean of vol[i-2]..vol[i-21] — N-1 candle
    # is deliberately excluded from its own mean, matching the live formula
    # vol_series[-22:-2] and the blueprint Section 5 English spec.
    vol_mean_n1 = df5["volume"].shift(2).rolling(20).mean()
    df5["volume_ratio_n1"] = df5["volume"].shift(1) / vol_mean_n1
    # volume_ratio_n2: N-2 volume divided by rolling mean of vol[i-3]..vol[i-22]
    vol_mean_n2 = df5["volume"].shift(3).rolling(20).mean()
    df5["volume_ratio_n2"] = df5["volume"].shift(2) / vol_mean_n2

    # ts_n1 = N-1 timestamp (shift by 1 for all multi-tf merges)
    ts_n1 = df5["timestamp"].shift(1)

    # -----------------------------------------------------------------------
    # 15m features — merge_asof backward on ts_n1
    # -----------------------------------------------------------------------
    atr15 = compute_atr14(df15)
    # CRITICAL (causality):
    # ccxt candle timestamps are candle OPEN times. At 12:50 (5m ts_n1), the
    # 15m candle stamped 12:45 is still forming until 13:00 in live trading.
    # If we merge unshifted 15m features on ts_n1<=12:45, training sees the
    # FINAL 12:45-13:00 values (future information), inflating backtest WR and
    # creating probability mismatch vs live. Shift(1) ensures we only expose
    # the most recently CLOSED 15m candle at every 5m row.
    df15["body_ratio_15m"] = ((df15["close"] - df15["open"]) / atr15).shift(1)
    df15["dir_15m"] = np.sign(df15["close"] - df15["open"]).shift(1)
    df15["volume_ratio_15m"] = (
        (df15["volume"] / df15["volume"].rolling(20, min_periods=2).mean()).shift(1)
    )

    r15 = _asof_backward(ts_n1, df15, ["body_ratio_15m", "dir_15m", "volume_ratio_15m"])
    df5["body_ratio_15m"] = r15["body_ratio_15m"].values
    df5["dir_15m"] = r15["dir_15m"].values
    df5["volume_ratio_15m"] = r15["volume_ratio_15m"].values

    # -----------------------------------------------------------------------
    # 1h features — same merge_asof pattern
    # -----------------------------------------------------------------------
    atr1h = compute_atr14(df1h)
    # Same causality guard as 15m: 1h timestamp is candle OPEN time, so the
    # candle stamped at hour H is not closed until H+1h. Shift all 1h features
    # by one bar so each 5m row only sees fully closed hourly context.
    df1h["body_ratio_1h"] = ((df1h["close"] - df1h["open"]) / atr1h).shift(1)
    df1h["dir_1h"] = np.sign(df1h["close"] - df1h["open"]).shift(1)
    ema9 = df1h["close"].ewm(span=9, adjust=False).mean()
    df1h["ema9_slope_1h"] = ((ema9 - ema9.shift(1)) / atr1h).shift(1)

    r1h = _asof_backward(ts_n1, df1h, ["body_ratio_1h", "dir_1h", "ema9_slope_1h"])
    df5["body_ratio_1h"] = r1h["body_ratio_1h"].values
    df5["dir_1h"] = r1h["dir_1h"].values
    df5["ema9_slope_1h"] = r1h["ema9_slope_1h"].values

    # -----------------------------------------------------------------------
    # Funding features
    # -----------------------------------------------------------------------
    funding["funding_zscore"] = (
        funding["funding_rate"] - funding["funding_rate"].rolling(24, min_periods=2).mean()
    ) / funding["funding_rate"].rolling(24, min_periods=2).std()
    funding.loc[funding["funding_rate"].rolling(24, min_periods=2).std() == 0, "funding_zscore"] = np.nan

    rf = _asof_backward(ts_n1, funding, ["funding_rate", "funding_zscore"])
    df5["funding_rate"] = rf["funding_rate"].values
    df5["funding_zscore"] = rf["funding_zscore"].values

    # -----------------------------------------------------------------------
    # OHLCV-native pressure features — computed purely from df5, zero parity gap
    # -----------------------------------------------------------------------
    hl_range = (df5["high"] - df5["low"]).clip(lower=1e-9)
    body      = df5["close"] - df5["open"]

    # body_ratio: candle body direction and strength, [-1, 1]
    df5["body_ratio"] = (body / hl_range).clip(-1.0, 1.0).shift(1)

    # upper_wick_ratio: selling rejection at highs, [0, 1]
    upper_wick = df5["high"] - df5[["open", "close"]].max(axis=1)
    df5["upper_wick_ratio"] = (upper_wick / hl_range).clip(0.0, 1.0).shift(1)

    # lower_wick_ratio: buying rejection at lows, [0, 1]
    lower_wick = df5[["open", "close"]].min(axis=1) - df5["low"]
    df5["lower_wick_ratio"] = (lower_wick / hl_range).clip(0.0, 1.0).shift(1)

    # vol_zscore: volume surge detection vs 20-bar rolling mean/std
    vol_mean20 = df5["volume"].rolling(20).mean()
    vol_std20  = df5["volume"].rolling(20).std(ddof=1).clip(lower=1e-8)
    df5["vol_zscore"] = ((df5["volume"] - vol_mean20) / vol_std20).shift(1)

    # vol_trend: short vs long volume momentum (5-bar / 20-bar rolling mean)
    vol_ma5  = df5["volume"].rolling(5).mean()
    vol_ma20 = df5["volume"].rolling(20).mean().clip(lower=1e-8)
    df5["vol_trend"] = (vol_ma5 / vol_ma20).shift(1)

    # -----------------------------------------------------------------------
    # Time-of-day cyclical features — derived from N-1 candle timestamp
    # Replaces raw hour_utc and dow with sine/cosine encoding so the model
    # can learn periodic patterns without discontinuities at midnight / week-end.
    # -----------------------------------------------------------------------
    ts_n1_series = df5["timestamp"].shift(1)
    hour_raw = ts_n1_series.dt.hour
    dow_raw = ts_n1_series.dt.dayofweek
    df5["hour_sin"] = np.sin(2 * np.pi * hour_raw / 24)
    df5["hour_cos"] = np.cos(2 * np.pi * hour_raw / 24)
    df5["dow_sin"]  = np.sin(2 * np.pi * dow_raw / 7)
    df5["dow_cos"]  = np.cos(2 * np.pi * dow_raw / 7)

    # Volatility regime features — derived from ATR of the N-1 candle
    # atr_percentile_24h: percentile rank (0.0–1.0) of atr5[i-1] within a 288-candle rolling window
    # vol_regime: zscore of atr5[i-1] within same 288-candle rolling window (std-normalized)
    # 288 = 24 hours * 12 five-minute candles per hour
    _ATR_WINDOW = 288
    atr_shifted = atr5.shift(1)
    def _rolling_percentile(s: pd.Series, w: int) -> pd.Series:
        # min_periods=14 allows partial windows during ATR warmup rows;
        # NaNs within the window are stripped before ranking so warmup NaNs
        # don't silently corrupt the percentile (NaN < value == False in numpy).
        def _pct(x: np.ndarray) -> float:
            x = x[~np.isnan(x)]
            if len(x) < 2:
                return np.nan
            return float(np.sum(x[:-1] < x[-1])) / max(len(x) - 1, 1)
        return s.rolling(w, min_periods=14).apply(_pct, raw=True)
    df5["atr_percentile_24h"] = _rolling_percentile(atr_shifted, _ATR_WINDOW)
    roll = atr_shifted.rolling(_ATR_WINDOW, min_periods=14)
    atr_roll_mean = roll.mean()
    atr_roll_std  = roll.std()
    df5["vol_regime"] = (atr_shifted - atr_roll_mean) / atr_roll_std.clip(lower=1e-10)

    # -----------------------------------------------------------------------
    # Momentum features (new) — all use shift(k>=1) for zero lookahead
    # -----------------------------------------------------------------------

    # rsi14: Wilder's RSI(14) on 5m closes, N-1 value
    _delta = df5["close"].diff()
    _gain = _delta.clip(lower=0)
    _loss = (-_delta).clip(lower=0)
    _avg_gain = _gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    _avg_loss = _loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    _rs = _avg_gain / _avg_loss.clip(lower=1e-10)
    _rsi = 100.0 - (100.0 / (1.0 + _rs))
    df5["rsi14"] = _rsi.shift(1)  # N-1 value, zero lookahead

    # candle_streak: consecutive same-direction candles ending at N-1
    # Vectorized approach: group consecutive same-direction runs
    _direction = np.sign(df5["close"] - df5["open"])
    _same_as_prev = (_direction == _direction.shift(1)) & (_direction != 0)
    _streak = _same_as_prev.groupby((~_same_as_prev).cumsum()).cumsum()
    _streak = _streak.where(_direction != 0, 0).astype(float)
    df5["candle_streak"] = _streak.shift(1)  # N-1 value

    # price_in_range: where N-1 close sits within 20-candle range ending at N-1
    # rolling(20).max/min on shift(1) gives range of [i-20..i-1] — zero lookahead
    _rolling_high = df5["high"].shift(1).rolling(20, min_periods=5).max()
    _rolling_low  = df5["low"].shift(1).rolling(20, min_periods=5).min()
    _rng = (_rolling_high - _rolling_low).clip(lower=1e-10)
    df5["price_in_range"] = (df5["close"].shift(1) - _rolling_low) / _rng

    # ema_cross_5m: sign of EMA9 vs EMA21 at N-1 candle (-1, 0, +1)
    _ema9_5m  = df5["close"].ewm(span=9,  adjust=False).mean()
    _ema21_5m = df5["close"].ewm(span=21, adjust=False).mean()
    df5["ema_cross_5m"] = np.sign(_ema9_5m - _ema21_5m).shift(1)  # N-1 cross state

    # -----------------------------------------------------------------------
    # Structure features — all use shift(k>=1) for zero lookahead
    # -----------------------------------------------------------------------

    # body_vs_range5: |body_n1| normalised by the 5-candle range ending at N-1.
    # 5-bar range = max(high[i-1..i-5]) - min(low[i-1..i-5]) — shift(1).rolling(5)
    # gives exactly that window at each row i with zero lookahead.
    _5bar_high   = df5["high"].shift(1).rolling(5, min_periods=2).max()
    _5bar_low    = df5["low"].shift(1).rolling(5, min_periods=2).min()
    _5bar_range  = (_5bar_high - _5bar_low).clip(lower=1e-9)
    df5["body_vs_range5"] = (df5["close"].shift(1) - df5["open"].shift(1)).abs() / _5bar_range

    # range_expansion: current 5-bar range vs prior 5-bar range (6..10 candles back).
    # shift(6).rolling(5) at row i = max/min of [i-6..i-10] — no overlap with current.
    _prior_high  = df5["high"].shift(6).rolling(5, min_periods=2).max()
    _prior_low   = df5["low"].shift(6).rolling(5, min_periods=2).min()
    _prior_range = (_prior_high - _prior_low).clip(lower=1e-9)
    df5["range_expansion"] = _5bar_range / _prior_range

    # vwap_dist_20: (close_n1 - vwap_20) / atr5_n1.
    # VWAP over the 20 candles ending at N-1: shift(1) before rolling ensures
    # the window [i-1..i-20] — zero lookahead. Divided by ATR for scale invariance.
    _cv_20  = (df5["close"].shift(1) * df5["volume"].shift(1)).rolling(20, min_periods=5).sum()
    _v_20   = df5["volume"].shift(1).rolling(20, min_periods=5).sum().clip(lower=1e-9)
    _vwap20 = _cv_20 / _v_20
    df5["vwap_dist_20"] = (df5["close"].shift(1) - _vwap20) / atr5.shift(1).clip(lower=1e-9)

    # -----------------------------------------------------------------------
    # Gate.io CVD taker flow features — merge_asof backward on ts_n1
    #
    # cvd_ratio     : long_taker_size / total_taker_size — [0, 1]
    #                 > 0.5 = net buy pressure, < 0.5 = net sell pressure.
    #                 Neutral default 0.5 when no CVD data is available.
    #
    # cvd_delta_norm: (long_taker_size - short_taker_size) / atr5 — signed,
    #                 ATR-normalized bar delta. Positive = buy dominance.
    #                 Neutral default 0.0 when no CVD data is available.
    #
    # Both use ts_n1 (N-1 candle timestamp) in the merge — zero lookahead.
    # -----------------------------------------------------------------------
    cvd_available = (
        cvd is not None
        and not cvd.empty
        and "long_taker_size" in cvd.columns
        and "short_taker_size" in cvd.columns
    )

    if cvd_available:
        cvd_clean = cvd.copy().reset_index(drop=True)
        cvd_clean["timestamp"] = cvd_clean["timestamp"].astype("datetime64[ms, UTC]")
        cvd_clean = cvd_clean.sort_values("timestamp").reset_index(drop=True)

        # Derive ratio and delta directly on the CVD frame before merging
        _total = (cvd_clean["long_taker_size"] + cvd_clean["short_taker_size"]).clip(lower=1e-9)
        cvd_clean["cvd_ratio"] = (cvd_clean["long_taker_size"] / _total).clip(0.0, 1.0)
        cvd_clean["cvd_delta"] = cvd_clean["long_taker_size"] - cvd_clean["short_taker_size"]

        # -----------------------------------------------------------------------
        # CVD accumulation + OI features — computed on the CVD frame (per-bar),
        # then merged onto df5 via _asof_backward exactly like cvd_ratio/delta.
        #
        # ALL use shift(k>=1) relative to the CVD frame so that when they are
        # merged onto df5 at ts_n1 (N-1 bar timestamp) the values they carry
        # correspond to history BEFORE N-1 — zero lookahead bias guaranteed.
        #
        # cvd_cumulative_5:  sum of cvd_delta over bars [t-5 .. t-1] (rolling 5,
        #                    then shift 1), ATR-normalized.  Captures short-term
        #                    directional buy/sell pressure accumulation.
        #
        # cvd_cumulative_20: same over bars [t-20 .. t-1] (rolling 20, shift 1).
        #                    Medium-term flow regime.
        #
        # cvd_trend_slope:   Linear regression slope of cvd_delta over the 10-bar
        #                    window [t-10 .. t-1] (rolling 10, shift 1), normalized
        #                    by ATR.  Positive = buy pressure accelerating.
        #                    Uses least-squares via polyfit on positions [0..9].
        #
        # cvd_divergence:    +1 if price direction (last 5 bars ending at t-1) and
        #                    CVD direction (last 5 bars ending at t-1) disagree,
        #                    -1 if they agree, 0 if either is flat.
        #                    Classical divergence: price up + CVD falling = reversal.
        #
        # oi_change_5bar:    (oi[t-1] - oi[t-6]) / |oi[t-6]|  — percentage change
        #                    in open interest over the prior 5 bars.  Positive =
        #                    new positions opening (trend continuation bias).
        #                    oi[t-6] is shift(6) so oi[t-1] = shift(1); diff is
        #                    (shift(1) - shift(6)) / abs(shift(6)).
        # -----------------------------------------------------------------------

        # ATR proxy for CVD normalization: we need ATR at the CVD bar's own
        # timestamp. We merge the ATR series from df5 onto cvd_clean via
        # _asof_backward so each CVD bar gets the ATR of the corresponding
        # 5m candle. Then we use that for normalization — not a fixed scalar.
        _df5_atr = pd.DataFrame({
            "timestamp": df5["timestamp"].values,
            "atr5_for_cvd": atr5.values,
        })
        _atr_merged = _asof_backward(
            cvd_clean["timestamp"], _df5_atr, ["atr5_for_cvd"]
        )
        _cvd_atr = _atr_merged["atr5_for_cvd"].clip(lower=1e-9)
        # _atr_s1: ATR at the bar BEFORE each CVD bar (shift(1) with proper index).
        # Using the Series directly (not wrapping in pd.Series()) preserves the
        # RangeIndex alignment with cvd_clean and eliminates any positional mismatch.
        _atr_s1 = _cvd_atr.shift(1)

        # cvd_cumulative_5: rolling sum of delta over 5 bars, then shift(1)
        # shift(1) moves the window one bar back so the value at CVD bar t represents
        # the sum of bars [t-5 .. t-1] — matching the live path which uses bars ending
        # at N-2 (one bar before the N-1 bar used as the merge anchor).
        cvd_clean["cvd_cumulative_5"] = (
            cvd_clean["cvd_delta"].rolling(5, min_periods=2).sum().shift(1)
            / _atr_s1.clip(lower=1e-9)
        )

        # cvd_cumulative_20: rolling sum of delta over 20 bars, then shift(1)
        cvd_clean["cvd_cumulative_20"] = (
            cvd_clean["cvd_delta"].rolling(20, min_periods=5).sum().shift(1)
            / _atr_s1.clip(lower=1e-9)
        )

        # cvd_trend_slope: OLS slope of cvd_delta over rolling 10-bar window, shift(1)
        # polyfit on positions [0..9] — normalized by ATR
        _x_slope = np.arange(10, dtype=np.float64)

        def _slope(vals: np.ndarray) -> float:
            """Return OLS slope of vals over positions 0..len-1."""
            v = vals[~np.isnan(vals)]
            if len(v) < 3:
                return np.nan
            x = np.arange(len(v), dtype=np.float64)
            try:
                return float(np.polyfit(x, v, 1)[0])
            except Exception:
                return np.nan

        cvd_clean["cvd_trend_slope"] = (
            cvd_clean["cvd_delta"]
            .rolling(10, min_periods=3)
            .apply(_slope, raw=True)
            .shift(1)
            / _atr_s1.clip(lower=1e-9)
        )

        # cvd_divergence: sign disagreement between price direction and CVD direction
        # price_dir_5: sign of sum of (close-open) over last 5 bars ending at t-1.
        # We pre-shift body5_sum in df5 coordinates BEFORE merging so that after
        # _asof_backward the merged value at each CVD timestamp already reflects
        # the price-body sum ending one 5m candle before that CVD bar.
        # This is equivalent to (and consistent with) the live path which reads
        # df5[-6:-1] (5 closed candles ending at N-1, not including forming bar N).
        _df5_body = pd.DataFrame({
            "timestamp": df5["timestamp"].values,
            "body5_sum": (df5["close"] - df5["open"]).rolling(5, min_periods=2).sum().shift(1).values,
        })
        _body_merged = _asof_backward(cvd_clean["timestamp"], _df5_body, ["body5_sum"])
        _price_dir_5 = np.sign(_body_merged["body5_sum"].values)
        _cvd_dir_5 = np.sign(
            cvd_clean["cvd_delta"].rolling(5, min_periods=2).sum().shift(1).values
        )
        # +1 = diverging (disagree), -1 = aligning (agree), 0 = either flat
        _div_raw = np.where(
            (_price_dir_5 == 0) | (_cvd_dir_5 == 0),
            0.0,
            np.where(_price_dir_5 != _cvd_dir_5, 1.0, -1.0),
        )
        cvd_clean["cvd_divergence"] = _div_raw

        # oi_change_5bar: (oi_t-1 - oi_t-6) / |oi_t-6|
        # open_interest column present when cvd_available (fetched from Gate.io)
        _oi_col = "open_interest"
        if _oi_col in cvd_clean.columns:
            _oi_s1 = cvd_clean[_oi_col].shift(1)
            _oi_s6 = cvd_clean[_oi_col].shift(6)
            cvd_clean["oi_change_5bar"] = (
                (_oi_s1 - _oi_s6) / _oi_s6.abs().clip(lower=1e-9)
            )
        else:
            # Missing open interest should not invalidate all CVD-enabled rows.
            # Use the documented neutral default, matching the no-CVD fallback.
            cvd_clean["oi_change_5bar"] = 0.0

        _cvd_cols = [
            "cvd_ratio", "cvd_delta",
            "cvd_cumulative_5", "cvd_cumulative_20", "cvd_trend_slope",
            "cvd_divergence", "oi_change_5bar",
        ]
        rcvd = _asof_backward(ts_n1, cvd_clean, _cvd_cols)
        df5["cvd_ratio"] = rcvd["cvd_ratio"].values

        # Normalize delta by ATR — use atr5 (already computed on df5)
        # Clip denominator away from zero to prevent inf/nan
        df5["cvd_delta_norm"] = rcvd["cvd_delta"].values / atr5.shift(1).clip(lower=1e-9).values

        # Accumulation + OI features are already ATR-normalized on the CVD frame
        df5["cvd_cumulative_5"]  = rcvd["cvd_cumulative_5"].values
        df5["cvd_cumulative_20"] = rcvd["cvd_cumulative_20"].values
        df5["cvd_trend_slope"]   = rcvd["cvd_trend_slope"].values
        df5["cvd_divergence"]    = rcvd["cvd_divergence"].values
        df5["oi_change_5bar"]    = rcvd["oi_change_5bar"].values
    else:
        log.warning(
            "build_features: CVD data not provided or empty — "
            "using neutral defaults (cvd_ratio=0.5, cvd_delta_norm=0.0, "
            "cvd_cumulative_5=0.0, cvd_cumulative_20=0.0, cvd_trend_slope=0.0, "
            "cvd_divergence=0.0, oi_change_5bar=0.0)"
        )
        df5["cvd_ratio"] = 0.5
        df5["cvd_delta_norm"] = 0.0
        df5["cvd_cumulative_5"] = 0.0
        df5["cvd_cumulative_20"] = 0.0
        df5["cvd_trend_slope"] = 0.0
        df5["cvd_divergence"] = 0.0
        df5["oi_change_5bar"] = 0.0

    # -----------------------------------------------------------------------
    # Target: 1 if next candle closes >= its own open (future label, NOT a feature)
    # Matches Polymarket settlement: close >= open within candle i+1
    # (resolver.py: winner = "Up" if close_price >= open_price else "Down")
    # -----------------------------------------------------------------------
    df5["target"] = (df5["close"].shift(-1) >= df5["open"].shift(-1)).astype(int)

    # -----------------------------------------------------------------------
    # Drop rows with any NaN in features or target, return feature cols + target
    # -----------------------------------------------------------------------
    all_cols = FEATURE_COLS + ["target"]
    df_out = df5[["timestamp"] + all_cols].dropna(subset=all_cols)
    log.info("build_features: %d rows after dropna (started with %d)", len(df_out), len(df5))
    return df_out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live feature computation
# ---------------------------------------------------------------------------

def _validate_live_inputs(
    df5_live: pd.DataFrame,
    df15_live: pd.DataFrame,
    df1h_live: pd.DataFrame,
) -> tuple[bool, str | None]:
    """Basic structural validation before canonical live feature construction."""
    required_ohlcv = {"timestamp", "open", "high", "low", "close", "volume"}
    frames = {
        "df5_live": df5_live,
        "df15_live": df15_live,
        "df1h_live": df1h_live,
    }

    for name, frame in frames.items():
        if frame is None or frame.empty:
            return False, f"{name} is empty"
        missing = sorted(required_ohlcv - set(frame.columns))
        if missing:
            return False, f"{name} missing columns: {', '.join(missing)}"

    return True, None


def _build_live_funding_frame(
    df5_live: pd.DataFrame,
    funding_rate_float: float | None,
    funding_records: deque,
) -> pd.DataFrame:
    """Materialize a live funding frame using real settlement timestamps.

    Preferred input is a deque of dict records with explicit settlement
    timestamps. For backward compatibility during the parity patch integration,
    callers that still pass a bare float deque are also accepted and converted
    into a minimally viable synthetic history anchored to the latest known
    settlement boundary.
    """
    timestamped_records: list[dict[str, object]] = []
    bare_rates: list[float] = []

    for item in funding_records:
        if isinstance(item, dict) and "timestamp" in item and "funding_rate" in item:
            ts = pd.to_datetime(item["timestamp"], utc=True)
            rate = float(item["funding_rate"])
            if pd.notna(ts) and pd.notna(rate):
                timestamped_records.append({"timestamp": ts, "funding_rate": rate})
        else:
            try:
                rate = float(item)
            except (TypeError, ValueError):
                continue
            if pd.notna(rate):
                bare_rates.append(rate)

    if timestamped_records:
        funding_df = pd.DataFrame(timestamped_records)
        funding_df = funding_df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
        return funding_df

    if bare_rates:
        last_ts = pd.Timestamp(df5_live["timestamp"].iloc[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        settlement_hour = (last_ts.hour // 8) * 8
        last_settlement = last_ts.replace(hour=settlement_hour, minute=0, second=0, microsecond=0)
        timestamps = pd.date_range(end=last_settlement, periods=len(bare_rates), freq="8h", tz="UTC")
        return pd.DataFrame({
            "timestamp": timestamps,
            "funding_rate": bare_rates,
        })

    if funding_rate_float is not None and pd.notna(funding_rate_float):
        last_ts = pd.Timestamp(df5_live["timestamp"].iloc[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        settlement_hour = (last_ts.hour // 8) * 8
        last_settlement = last_ts.replace(hour=settlement_hour, minute=0, second=0, microsecond=0)
        return pd.DataFrame({
            "timestamp": [last_settlement],
            "funding_rate": [float(funding_rate_float)],
        })

    return pd.DataFrame(columns=["timestamp", "funding_rate"])


def build_live_features(
    df5_live: pd.DataFrame,
    df15_live: pd.DataFrame,
    df1h_live: pd.DataFrame,
    funding_rate_float: float | None,
    funding_records: deque,
    cvd_live: pd.DataFrame | None = None,
) -> "tuple[np.ndarray, list[str]] | tuple[None, list[str]]":
    """Build the live feature row via the canonical historical feature pipeline.

    The live path intentionally avoids a separate hand-maintained feature
    implementation. Instead it reconstructs the minimal live inputs needed by
    build_features(), runs the same canonical pipeline used in training, and
    extracts the final inference row from that output.

    Contract:
      - df5_live includes the still-forming 5m candle as the last row.
      - The returned row corresponds to the latest fully closed 5m candle and is
        therefore aligned with the N+1 prediction contract used in training.
      - Returns (None, [...]) on validation or feature-construction failure,
        with explicit feature names when NaNs are present.
    """
    ok, validation_error = _validate_live_inputs(df5_live, df15_live, df1h_live)
    if not ok:
        log.warning("build_live_features: %s", validation_error)
        return None, []

    try:
        df5 = df5_live.copy().reset_index(drop=True)
        df15 = df15_live.copy().reset_index(drop=True)
        df1h = df1h_live.copy().reset_index(drop=True)

        for frame_name, frame in (("df5", df5), ("df15", df15), ("df1h", df1h)):
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).astype("datetime64[ms, UTC]")
            frame.sort_values("timestamp", inplace=True)
            frame.reset_index(drop=True, inplace=True)
            if frame["timestamp"].duplicated().any():
                dupes = int(frame["timestamp"].duplicated().sum())
                log.warning(
                    "build_live_features: %s contains %d duplicate timestamps; keeping last occurrence",
                    frame_name,
                    dupes,
                )
                frame.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
                frame.reset_index(drop=True, inplace=True)

        funding_df = _build_live_funding_frame(df5, funding_rate_float, funding_records)

        cvd_df = None
        if cvd_live is not None and not cvd_live.empty:
            cvd_df = cvd_live.copy().reset_index(drop=True)
            if "timestamp" in cvd_df.columns:
                cvd_df["timestamp"] = pd.to_datetime(cvd_df["timestamp"], utc=True).astype("datetime64[ms, UTC]")
                cvd_df.sort_values("timestamp", inplace=True)
                cvd_df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
                cvd_df.reset_index(drop=True, inplace=True)

        feat_df = build_features(df5, df15, df1h, funding_df, cvd_df)
        if feat_df.empty:
            log.warning("build_live_features: canonical build_features() returned no usable rows")
            return None, []

        latest = feat_df.iloc[-1]
        row = latest[FEATURE_COLS].to_numpy(dtype=np.float64, copy=True).reshape(1, -1)
        nan_features = [FEATURE_COLS[i] for i, value in enumerate(row[0]) if np.isnan(value)]
        if nan_features:
            log.warning("build_live_features: NaN in canonical feature row, skipping inference. NaN features: %s", nan_features)
            return None, nan_features

        return row, []

    except Exception:
        log.exception("build_live_features: canonical live feature construction failed")
        return None, []
