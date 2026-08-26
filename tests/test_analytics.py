"""Sharpe / Sortino / max DD / origin labeling. Zero vol -> nan."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl

from qre.analytics.metrics import (
    PerformanceReport,
    annualized_vol,
    compute,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from qre.data.loader import generate_synthetic_ohlcv
from qre.execution.model import ExecutionModel
from qre.features.pipeline import build_features
from qre.portfolio.simulator import SimulationResult, simulate
from qre.strategies.momentum import TimeSeriesMomentum
from qre.types import TRADING_DAYS_PER_YEAR, DataOrigin


def test_sharpe_formula_and_zero_vol_nan() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.01, size=252)
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    expected = mu / sd * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert abs(sharpe_ratio(r) - expected) < 1e-12
    assert math.isnan(sharpe_ratio(np.zeros(50)))
    assert math.isnan(annualized_vol(np.zeros(50)))


def test_sortino_downside_deviation() -> None:
    r = np.array([0.02, -0.01, 0.00, -0.03, 0.01], dtype=float)
    mu = float(np.mean(r))
    downside = np.minimum(r, 0.0)
    dd = float(np.sqrt(np.mean(downside * downside)))
    expected = mu / dd * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert abs(sortino_ratio(r) - expected) < 1e-12
    assert math.isnan(sortino_ratio(np.array([0.01, 0.02, 0.00])))


def test_max_drawdown_on_equity() -> None:
    eq = np.array([100.0, 110.0, 105.0, 90.0, 95.0])
    assert abs(max_drawdown(eq) - (90.0 / 110.0 - 1.0)) < 1e-12


def test_report_carries_origin_and_after_costs() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=3, n_bars=80, seed=31, process="gbm")
    feat = build_features(panel, lookback=8)
    w = TimeSeriesMomentum(lookback=8).generate_weights(feat)
    result = simulate(panel, w, ExecutionModel())
    report = compute(result, after_costs=True)
    assert isinstance(report, PerformanceReport)
    assert report.data_origin is DataOrigin.SYNTHETIC
    assert report.after_costs is True
    d = report.as_dict()
    assert d["data_origin"] == "SYNTHETIC"
    assert d["after_costs"] is True


def test_empty_sim_result_struct() -> None:
    start = date(2020, 1, 2)
    eq = pl.DataFrame(
        {
            "date": [start, start + timedelta(days=1)],
            "nav": [1_000_000.0, 1_000_000.0],
            "gross_ret": [0.0, 0.0],
            "net_ret": [0.0, 0.0],
            "cost": [0.0, 0.0],
            "turnover": [0.0, 0.0],
            "gross_exposure": [0.0, 0.0],
            "net_exposure": [0.0, 0.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    result = SimulationResult(
        equity=eq,
        holdings=pl.DataFrame({"date": [], "symbol": [], "weight": []}),
        data_origin=DataOrigin.SYNTHETIC,
        after_costs=True,
        initial_nav=1_000_000.0,
        fill_delay=1,
    )
    report = compute(result)
    assert report.data_origin is DataOrigin.SYNTHETIC
    assert math.isnan(report.sharpe)
