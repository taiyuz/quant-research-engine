"""Causal return transforms. Rolling windows use prices at dates <= t only."""

from __future__ import annotations

import polars as pl


def add_simple_return(frame: pl.DataFrame, price_col: str = "close") -> pl.DataFrame:
    """r_t = close_t / close_{t-1} - 1, grouped by symbol. Null on the first bar."""
    frame = frame.sort(["symbol", "date"])
    return frame.with_columns(
        (pl.col(price_col) / pl.col(price_col).shift(1).over("symbol") - 1.0).alias(
            "simple_return"
        )
    )


def add_lookback_return(
    frame: pl.DataFrame, lookback: int, price_col: str = "close"
) -> pl.DataFrame:
    """close_t / close_{t-lookback} - 1. Uses only prices at or before t."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    frame = frame.sort(["symbol", "date"])
    return frame.with_columns(
        (pl.col(price_col) / pl.col(price_col).shift(lookback).over("symbol") - 1.0).alias(
            f"ret_{lookback}"
        )
    )


def add_log_price(frame: pl.DataFrame, price_col: str = "close") -> pl.DataFrame:
    frame = frame.sort(["symbol", "date"])
    return frame.with_columns(pl.col(price_col).log().alias("log_close"))
