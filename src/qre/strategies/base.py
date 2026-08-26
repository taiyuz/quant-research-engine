"""Strategy protocol: map causal features to target weights.

Weights dated t are a function of information through t. They are filled
on the next bar (see execution model). Gross exposure is typically scaled
to 1.0 when there is a valid signal; names with null features get 0.
"""

from __future__ import annotations

from typing import Protocol

import polars as pl

WEIGHT_COLUMNS = ("date", "symbol", "target_weight")


class Strategy(Protocol):
    name: str

    def generate_weights(self, features: pl.DataFrame) -> pl.DataFrame:
        """Return date, symbol, target_weight. One row per (date, symbol) in features."""
        ...


def _require_cols(frame: pl.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"features missing columns {missing}")


def _finalize_weights(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.select(
            pl.col("date"),
            pl.col("symbol"),
            pl.col("target_weight").fill_null(0.0).cast(pl.Float64),
        )
        .sort(["date", "symbol"])
    )


def scale_gross(frame: pl.DataFrame, gross: float = 1.0) -> pl.DataFrame:
    """Scale per-date weights so sum(|w|) == gross when there is any exposure."""
    g = (
        frame.group_by("date")
        .agg(pl.col("target_weight").abs().sum().alias("_gross"))
    )
    out = frame.join(g, on="date")
    out = out.with_columns(
        pl.when(pl.col("_gross") <= 0.0)
        .then(0.0)
        .otherwise(pl.col("target_weight") * (gross / pl.col("_gross")))
        .alias("target_weight")
    ).drop("_gross")
    return out


def build_strategy(name: str, lookback: int, **kwargs: object) -> Strategy:
    from qre.strategies.cross_sectional import CrossSectionalRank
    from qre.strategies.mean_reversion import MeanReversion
    from qre.strategies.momentum import TimeSeriesMomentum
    from qre.strategies.pairs import PairsResidual

    key = name.strip().lower().replace("-", "_")
    if key in {"momentum", "ts_momentum", "time_series_momentum"}:
        return TimeSeriesMomentum(lookback=lookback)
    if key in {"mean_reversion", "mr", "zscore"}:
        return MeanReversion(lookback=lookback)
    if key in {"cross_sectional", "xs", "rank"}:
        n_long = int(kwargs.get("n_long", 0) or 0)
        return CrossSectionalRank(lookback=lookback, n_long=n_long or None)
    if key in {"pairs", "pair", "residual"}:
        lag = kwargs.get("leg_a")
        lbg = kwargs.get("leg_b")
        return PairsResidual(
            lookback=lookback,
            leg_a=str(lag) if lag else None,
            leg_b=str(lbg) if lbg else None,
        )
    raise ValueError(f"unknown strategy {name!r}")
