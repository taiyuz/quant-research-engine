"""Cross-sectional rank long-short, dollar-neutral.

On each date, rank names by lookback return. Long the top half (or n_long),
short the bottom half. Weights sum to 0 (dollar-neutral) and sum(|w|) = 1
among names with a valid rank. Ties are broken by symbol so ranking is
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qre.strategies.base import _finalize_weights, _require_cols


@dataclass(frozen=True)
class CrossSectionalRank:
    lookback: int
    n_long: int | None = None
    name: str = "cross_sectional"

    def generate_weights(self, features: pl.DataFrame) -> pl.DataFrame:
        col = f"ret_{self.lookback}"
        _require_cols(features, ("date", "symbol", col))
        ranked = (
            features.select("date", "symbol", pl.col(col).alias("score"))
            .filter(pl.col("score").is_not_null())
            .sort(["date", "score", "symbol"])
            .with_columns(
                pl.len().over("date").alias("n"),
                pl.col("score").rank(method="ordinal").over("date").alias("rk"),
            )
        )
        if self.n_long is None:
            long_mask = pl.col("rk") > (pl.col("n") / 2.0)
            short_mask = pl.col("rk") <= (pl.col("n") / 2.0)
        else:
            n_long = self.n_long
            long_mask = pl.col("rk") > (pl.col("n") - n_long)
            short_mask = pl.col("rk") <= n_long

        ranked = ranked.with_columns(
            pl.when(long_mask)
            .then(1.0)
            .when(short_mask)
            .then(-1.0)
            .otherwise(0.0)
            .alias("target_weight")
        )
        all_rows = features.select("date", "symbol")
        frame = all_rows.join(
            ranked.select("date", "symbol", "target_weight"), on=["date", "symbol"], how="left"
        )
        sides = frame.group_by("date").agg(
            pl.when(pl.col("target_weight") > 0).then(pl.col("target_weight")).otherwise(0.0).sum().alias("long_g"),
            pl.when(pl.col("target_weight") < 0).then(pl.col("target_weight").abs()).otherwise(0.0).sum().alias("short_g"),
        )
        frame = frame.join(sides, on="date").with_columns(
            pl.when((pl.col("long_g") > 0) & (pl.col("short_g") > 0) & (pl.col("target_weight") > 0))
            .then(pl.col("target_weight") / pl.col("long_g") * 0.5)
            .when((pl.col("long_g") > 0) & (pl.col("short_g") > 0) & (pl.col("target_weight") < 0))
            .then(pl.col("target_weight") / pl.col("short_g") * 0.5)
            .otherwise(pl.col("target_weight"))
            .alias("target_weight")
        ).drop(["long_g", "short_g"])
        return _finalize_weights(frame)
