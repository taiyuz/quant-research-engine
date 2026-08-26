"""Pairs: trade the residual / spread of two names.

Spread_t = log(close_A,t) - log(close_B,t)  (hedge ratio 1; synthetic pair
is generated cointegrated-like under process=ou_pair).
z_t is the rolling z-score of the spread using a window ending at t.
Position: short the rich leg, long the cheap leg, dollar-neutral, unit gross.

If legs are omitted, the first two symbols in lexicographic order are used
(SYN000/SYN001 from the synthetic generator).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qre.strategies.base import _finalize_weights, _require_cols


@dataclass(frozen=True)
class PairsResidual:
    lookback: int
    leg_a: str | None = None
    leg_b: str | None = None
    z_entry: float = 0.0
    name: str = "pairs"

    def generate_weights(self, features: pl.DataFrame) -> pl.DataFrame:
        _require_cols(features, ("date", "symbol", "log_close"))
        symbols = sorted(features.get_column("symbol").unique().to_list())
        if self.leg_a and self.leg_b:
            a, b = self.leg_a, self.leg_b
        else:
            if len(symbols) < 2:
                raise ValueError("pairs strategy needs at least two symbols")
            a, b = symbols[0], symbols[1]

        wide = (
            features.filter(pl.col("symbol").is_in([a, b]))
            .select("date", "symbol", "log_close")
            .pivot(values="log_close", index="date", on="symbol")
            .sort("date")
        )
        if a not in wide.columns or b not in wide.columns:
            raise ValueError(f"pair legs {a}/{b} not present in features")
        wide = wide.with_columns((pl.col(a) - pl.col(b)).alias("spread"))
        wide = wide.with_columns(
            pl.col("spread").rolling_mean(window_size=self.lookback, min_samples=self.lookback).alias("sp_ma"),
            pl.col("spread").rolling_std(window_size=self.lookback, min_samples=self.lookback).alias("sp_sd"),
        )
        wide = wide.with_columns(
            pl.when(pl.col("sp_sd").is_null() | (pl.col("sp_sd") == 0.0))
            .then(None)
            .otherwise((pl.col("spread") - pl.col("sp_ma")) / pl.col("sp_sd"))
            .alias("spread_z")
        )
        pos = wide.select(
            "date",
            pl.when(pl.col("spread_z").is_null())
            .then(0.0)
            .when(pl.col("spread_z").abs() < self.z_entry)
            .then(0.0)
            .otherwise(-pl.col("spread_z").sign())
            .alias("w_a"),
        )
        pos = pos.with_columns((-pl.col("w_a")).alias("w_b"))
        pos = pos.with_columns(
            pl.when(pl.col("w_a") == 0.0)
            .then(0.0)
            .otherwise(pl.col("w_a") * 0.5)
            .alias("w_a"),
            pl.when(pl.col("w_b") == 0.0)
            .then(0.0)
            .otherwise(pl.col("w_b") * 0.5)
            .alias("w_b"),
        )
        long_a = pos.select("date", pl.lit(a).alias("symbol"), pl.col("w_a").alias("target_weight"))
        long_b = pos.select("date", pl.lit(b).alias("symbol"), pl.col("w_b").alias("target_weight"))
        pair_w = pl.concat([long_a, long_b])
        all_rows = features.select("date", "symbol")
        frame = all_rows.join(pair_w, on=["date", "symbol"], how="left")
        return _finalize_weights(frame)
