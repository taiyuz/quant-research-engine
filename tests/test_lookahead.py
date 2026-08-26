"""Look-ahead: features at t must be invariant to prices after t."""

from __future__ import annotations

from datetime import date

import polars as pl

from qre.data.loader import generate_synthetic_ohlcv
from qre.features.pipeline import assert_features_invariant_to_future, build_features
from qre.types import BarPanel


def _features(panel: BarPanel) -> pl.DataFrame:
    return build_features(panel, lookback=10)


def test_features_match_when_only_future_prices_change() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=4, n_bars=80, seed=1, process="gbm")
    dates = panel.dates
    cutoff = dates[40]
    mutated = panel.frame.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("close") * 3.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    mutated = mutated.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("open") * 3.0)
        .otherwise(pl.col("open"))
        .alias("open"),
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("high") * 3.0)
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("low") * 3.0)
        .otherwise(pl.col("low"))
        .alias("low"),
    )
    other = BarPanel(frame=mutated, metadata=panel.metadata)
    a_after = panel.frame.filter(pl.col("date") > cutoff).get_column("close")
    b_after = other.frame.filter(pl.col("date") > cutoff).get_column("close")
    assert (a_after != b_after).any()
    assert_features_invariant_to_future(
        _features,
        panel,
        other,
        cutoff,
        ["simple_return", "ret_10", "zscore", "log_close"],
    )


def test_two_panels_identical_through_t_features_equal() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=3, n_bars=60, seed=2, process="ou")
    cutoff: date = panel.dates[25]
    through = panel.copy_with_frame(panel.frame.filter(pl.col("date") <= cutoff))
    feat_full = _features(panel).filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
    feat_trunc = _features(through).sort(["symbol", "date"])
    for col in ["simple_return", "ret_10", "zscore"]:
        left = feat_full.get_column(col)
        right = feat_trunc.get_column(col)
        both_null = left.is_null() & right.is_null()
        close = (left - right).abs() <= 1e-12
        assert bool((both_null | close).all()), col
