"""All four strategies produce weights with the intended signs and constraints."""

from __future__ import annotations

import polars as pl

from qre.data.loader import generate_synthetic_ohlcv
from qre.features.pipeline import build_features
from qre.strategies.base import build_strategy
from qre.strategies.cross_sectional import CrossSectionalRank
from qre.strategies.mean_reversion import MeanReversion
from qre.strategies.momentum import TimeSeriesMomentum
from qre.strategies.pairs import PairsResidual


def test_momentum_sign_of_lookback_return() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=4, n_bars=80, seed=21, process="gbm")
    feat = build_features(panel, lookback=10)
    w = TimeSeriesMomentum(lookback=10).generate_weights(feat)
    joined = feat.join(w, on=["date", "symbol"])
    valid = joined.filter(pl.col("ret_10").is_not_null() & (pl.col("ret_10") != 0))
    assert valid.height > 0
    assert bool(
        (
            (valid.get_column("target_weight") == 0)
            | (valid.get_column("target_weight").sign() == valid.get_column("ret_10").sign())
        ).all()
    )


def test_mean_reversion_fades_zscore() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=3, n_bars=80, seed=22, process="ou")
    feat = build_features(panel, lookback=10)
    w = MeanReversion(lookback=10).generate_weights(feat)
    joined = feat.join(w, on=["date", "symbol"])
    valid = joined.filter(pl.col("zscore").is_not_null() & (pl.col("zscore") != 0))
    assert valid.height > 0
    assert bool(
        (
            (valid.get_column("target_weight") == 0)
            | (valid.get_column("target_weight").sign() == -valid.get_column("zscore").sign())
        ).all()
    )


def test_cross_sectional_dollar_neutral() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=8, n_bars=80, seed=23, process="gbm")
    feat = build_features(panel, lookback=10)
    w = CrossSectionalRank(lookback=10).generate_weights(feat)
    g = w.group_by("date").agg(
        pl.col("target_weight").sum().alias("net"),
        pl.col("target_weight").abs().sum().alias("gross"),
    )
    live = g.filter(pl.col("gross") > 0)
    assert live.height > 0
    assert float(live.get_column("net").abs().max()) < 1e-10
    assert float((live.get_column("gross") - 1.0).abs().max()) < 1e-10


def test_pairs_opposite_legs() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=3, n_bars=120, seed=24, process="ou_pair")
    feat = build_features(panel, lookback=20)
    w = PairsResidual(lookback=20, leg_a="SYN000", leg_b="SYN001").generate_weights(feat)
    pair = w.filter(pl.col("symbol").is_in(["SYN000", "SYN001"]))
    wide = pair.pivot(values="target_weight", index="date", on="symbol").drop_nulls()
    live = wide.filter((pl.col("SYN000") != 0) | (pl.col("SYN001") != 0))
    assert live.height > 0
    net = live.get_column("SYN000") + live.get_column("SYN001")
    assert float(net.abs().max()) < 1e-12
    prod = live.get_column("SYN000") * live.get_column("SYN001")
    assert float(prod.max()) <= 0.0
    others = w.filter(~pl.col("symbol").is_in(["SYN000", "SYN001"]))
    assert float(others.get_column("target_weight").abs().sum()) == 0.0


def test_build_strategy_registry() -> None:
    assert build_strategy("momentum", 10).name == "momentum"
    assert build_strategy("mean_reversion", 10).name == "mean_reversion"
    assert build_strategy("cross_sectional", 10).name == "cross_sectional"
    assert build_strategy("pairs", 10).name == "pairs"
