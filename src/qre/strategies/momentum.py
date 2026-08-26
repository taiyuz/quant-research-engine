"""Time-series momentum: sign of lagged lookback return.

w_{i,t} proportional to sign(close_t / close_{t-lookback} - 1).
Equal-weighted across names that have a non-null lookback return.
This is not a claim that TSMOM works on the synthetic GBM panel (drift 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qre.strategies.base import _finalize_weights, _require_cols, scale_gross


@dataclass(frozen=True)
class TimeSeriesMomentum:
    lookback: int
    name: str = "momentum"

    def generate_weights(self, features: pl.DataFrame) -> pl.DataFrame:
        col = f"ret_{self.lookback}"
        _require_cols(features, ("date", "symbol", col))
        frame = features.select(
            "date",
            "symbol",
            pl.when(pl.col(col).is_null())
            .then(None)
            .otherwise(pl.col(col).sign())
            .alias("target_weight"),
        )
        frame = scale_gross(frame, gross=1.0)
        return _finalize_weights(frame)
