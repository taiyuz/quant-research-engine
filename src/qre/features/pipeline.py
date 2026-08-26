"""Lagged / rolling feature pipeline with look-ahead guards.

Convention (enforced by the simulator, documented here so it cannot drift):
- A feature row dated t may use close[t] (and any history <= t).
- It must be invariant to prices after t.
- The resulting signal is NOT tradable at t's close. Default execution is
  fill at t+1 close after costs. See qre.execution.model.

Rolling windows are right-aligned: the value at t is the statistic of
{t-window+1, ..., t}. Polars rolling_* is causal when data is sorted.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from qre.features.returns import add_log_price, add_lookback_return, add_simple_return
from qre.types import BarPanel


def add_rolling_mean(
    frame: pl.DataFrame, window: int, col: str = "close", alias: str | None = None
) -> pl.DataFrame:
    if window < 1:
        raise ValueError("window must be >= 1")
    name = alias or f"{col}_ma_{window}"
    frame = frame.sort(["symbol", "date"])
    return frame.with_columns(
        pl.col(col).rolling_mean(window_size=window, min_samples=window).over("symbol").alias(name)
    )


def add_rolling_std(
    frame: pl.DataFrame, window: int, col: str = "close", alias: str | None = None
) -> pl.DataFrame:
    if window < 1:
        raise ValueError("window must be >= 1")
    name = alias or f"{col}_sd_{window}"
    frame = frame.sort(["symbol", "date"])
    return frame.with_columns(
        pl.col(col).rolling_std(window_size=window, min_samples=window).over("symbol").alias(name)
    )


def add_rolling_zscore(
    frame: pl.DataFrame, window: int, col: str = "close", alias: str = "zscore"
) -> pl.DataFrame:
    """z_t = (x_t - MA_t) / SD_t with MA/SD computed on {t-window+1, ..., t}."""
    ma = f"{col}_ma_{window}"
    sd = f"{col}_sd_{window}"
    frame = add_rolling_mean(frame, window, col, alias=ma)
    frame = add_rolling_std(frame, window, col, alias=sd)
    return frame.with_columns(
        pl.when(pl.col(sd).is_null() | (pl.col(sd) == 0.0))
        .then(None)
        .otherwise((pl.col(col) - pl.col(ma)) / pl.col(sd))
        .alias(alias)
    )


def shift_for_execution(frame: pl.DataFrame, cols: list[str], bars: int = 1) -> pl.DataFrame:
    """Shift signal columns forward in event time so they cannot be used on the same bar."""
    if bars < 1:
        raise ValueError("bars must be >= 1")
    frame = frame.sort(["symbol", "date"])
    return frame.with_columns(
        [pl.col(c).shift(bars).over("symbol").alias(f"{c}_lag{bars}") for c in cols]
    )


def build_features(panel: BarPanel, lookback: int) -> pl.DataFrame:
    """Standard feature set used by the four bundled strategies.

    All rolling / lag operations are causal. lookback is both the momentum
    window and the z-score window.
    """
    frame = panel.frame.sort(["symbol", "date"])
    frame = add_simple_return(frame)
    frame = add_lookback_return(frame, lookback)
    frame = add_log_price(frame)
    frame = add_rolling_zscore(frame, lookback, col="close", alias="zscore")
    frame = add_rolling_zscore(frame, lookback, col="log_close", alias="log_zscore")
    return frame


def assert_features_invariant_to_future(
    feature_fn,
    panel_through_t: BarPanel,
    panel_with_future_mutated: BarPanel,
    cutoff: date,
    feature_cols: list[str],
) -> None:
    """Look-ahead assertion: two panels identical through t, different after,
    must produce identical feature values through t.
    """
    feat_a = feature_fn(panel_through_t)
    feat_b = feature_fn(panel_with_future_mutated)
    cols = ["date", "symbol", *feature_cols]
    a = feat_a.filter(pl.col("date") <= cutoff).select(cols).sort(["symbol", "date"])
    b = feat_b.filter(pl.col("date") <= cutoff).select(cols).sort(["symbol", "date"])
    if a.shape != b.shape:
        raise AssertionError(f"look-ahead shape mismatch: {a.shape} vs {b.shape}")
    joined = a.join(b, on=["date", "symbol"], how="inner", suffix="_mut")
    for col in feature_cols:
        left = joined.get_column(col)
        right = joined.get_column(f"{col}_mut")
        both_null = left.is_null() & right.is_null()
        close = (left - right).abs() <= 1e-12
        ok = both_null | close
        if not bool(ok.all()):
            n_bad = int((~ok).sum())
            raise AssertionError(
                f"look-ahead leak in {col}: {n_bad} rows through {cutoff} changed "
                "after mutating future prices"
            )
