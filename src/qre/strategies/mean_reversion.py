"""Mean reversion: fade the rolling z-score of price versus its moving average.

w_{i,t} proportional to -clip(z_t, -3, 3), then scaled to unit gross.
z_t uses a right-aligned window ending at t. Execution is still next bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qre.strategies.base import _finalize_weights, _require_cols, scale_gross


@dataclass(frozen=True)
class MeanReversion:
    lookback: int
    z_cap: float = 3.0
    name: str = "mean_reversion"

    def generate_weights(self, features: pl.DataFrame) -> pl.DataFrame:
        _require_cols(features, ("date", "symbol", "zscore"))
        z = pl.col("zscore").clip(-self.z_cap, self.z_cap)
        frame = features.select(
            "date",
            "symbol",
            pl.when(pl.col("zscore").is_null()).then(None).otherwise(-z).alias("target_weight"),
        )
        frame = scale_gross(frame, gross=1.0)
        return _finalize_weights(frame)
