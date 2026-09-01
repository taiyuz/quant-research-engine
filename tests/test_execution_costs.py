"""Gross PnL >= net PnL when turnover > 0. Cost formula is fee + slippage on notional."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from qre.analytics.metrics import compute
from qre.data.loader import generate_synthetic_ohlcv
from qre.execution.model import ExecutionModel
from qre.features.pipeline import build_features
from qre.portfolio.simulator import simulate
from qre.strategies.momentum import TimeSeriesMomentum
from qre.types import BarPanel, DataOrigin, PanelMetadata


def _panel_two_days() -> BarPanel:
    d0 = date(2020, 1, 2)
    d1 = date(2020, 1, 3)
    d2 = date(2020, 1, 6)
    rows = []
    for d, px in ((d0, 100.0), (d1, 110.0), (d2, 121.0)):
        rows.append(
            {
                "date": d,
                "symbol": "AAA",
                "open": px,
                "high": px,
                "low": px,
                "close": px,
                "volume": 1,
            }
        )
    return BarPanel(
        frame=pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)),
        metadata=PanelMetadata(origin=DataOrigin.SYNTHETIC, notes="SYNTHETIC fixture"),
    )


def _panel_uptrend(n_bars: int, start_px: float, daily_ret: float) -> BarPanel:
    """Labeled SYNTHETIC close-only path with a constant daily simple return."""
    start = date(2020, 1, 2)
    rows = []
    px = start_px
    for i in range(n_bars):
        d = start + timedelta(days=i)
        rows.append(
            {
                "date": d,
                "symbol": "AAA",
                "open": px,
                "high": px,
                "low": px,
                "close": px,
                "volume": 1,
            }
        )
        px *= 1.0 + daily_ret
    return BarPanel(
        frame=pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)),
        metadata=PanelMetadata(
            origin=DataOrigin.SYNTHETIC,
            notes="SYNTHETIC 5bp/day uptrend fixture — not a market",
        ),
    )


def test_cost_formula() -> None:
    m = ExecutionModel(commission_bps=1.0, slippage_bps=2.0)
    assert abs(m.cost_rate - 3.0 / 1e4) < 1e-15
    nav = 1_000_000.0
    abs_delta = 0.5
    cost = m.cost_from_weight_delta(abs_delta, nav)
    assert abs(cost - (3.0 / 1e4) * 0.5 * nav) < 1e-9


def test_fill_delay_rejects_same_bar() -> None:
    """Look-ahead at execution: fill_delay < 1 would trade the signal bar's close."""
    with pytest.raises(ValueError, match="fill_delay"):
        ExecutionModel(fill_delay=0)
    ok = ExecutionModel(fill_delay=1)
    assert ok.fill_delay == 1


def test_gross_pnl_ge_net_pnl_when_turnover() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=4, n_bars=120, seed=11, process="gbm")
    feat = build_features(panel, lookback=10)
    w = TimeSeriesMomentum(lookback=10).generate_weights(feat)
    result = simulate(panel, w, ExecutionModel(commission_bps=5.0, slippage_bps=5.0))
    eq = result.equity.filter(pl.col("turnover") > 0)
    assert eq.height > 0
    report = compute(result, after_costs=True)
    assert report.total_cost > 0
    assert report.gross_pnl >= report.net_pnl - 1e-6
    hit = result.equity.filter(pl.col("cost") > 0)
    assert hit.height > 0
    for row in hit.iter_rows(named=True):
        assert row["cost"] > 0
        assert row["net_ret"] <= row["gross_ret"] + 1e-12


def test_zero_turnover_zero_cost() -> None:
    panel = _panel_two_days()
    weights = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
            "symbol": ["AAA", "AAA", "AAA"],
            "target_weight": [0.0, 0.0, 0.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    result = simulate(panel, weights, ExecutionModel())
    assert float(result.equity.get_column("cost").sum()) == 0.0
    assert float(result.equity.get_column("turnover").sum()) == 0.0


def test_next_bar_fill_does_not_earn_same_bar_move() -> None:
    """Signal dated d0 filled at d1 close; the d0->d1 return is NOT captured by that signal."""
    panel = _panel_two_days()
    weights = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
            "symbol": ["AAA", "AAA", "AAA"],
            "target_weight": [1.0, 1.0, 0.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    result = simulate(
        panel, weights, ExecutionModel(commission_bps=0.0, slippage_bps=0.0), initial_nav=1000.0
    )
    eq = result.equity.sort("date")
    rets = eq.get_column("gross_ret").to_list()
    assert abs(rets[0]) < 1e-12
    assert abs(rets[1]) < 1e-12
    assert abs(rets[2] - (121.0 / 110.0 - 1.0)) < 1e-12


def test_high_turnover_costs_flip_modest_gross_profit() -> None:
    """100% daily turnover at 10 bps one-way can flip a small trend into a net loss.

    Sloppy backtests report the zero-cost (gross) number. RESEARCH.md §3: a
    strategy that turns over 100% a day at 10 bps round-trip needs a raw edge
    most textbook signals do not have. This fixture is labeled SYNTHETIC and
    is not a market result; it only asserts the sign flip. No Sharpe is claimed.
    """
    daily_ret = 0.0005  # 5 bp/day drift, below 10 bp one-way cost
    panel = _panel_uptrend(n_bars=20, start_px=100.0, daily_ret=daily_ret)
    dates = panel.dates
    weights = pl.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * len(dates),
            "target_weight": [1.0 if i % 2 else 0.0 for i in range(len(dates))],
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    zero = simulate(
        panel,
        weights,
        ExecutionModel(commission_bps=0.0, slippage_bps=0.0),
        initial_nav=1_000_000.0,
    )
    taxed = simulate(
        panel,
        weights,
        ExecutionModel(commission_bps=5.0, slippage_bps=5.0),
        initial_nav=1_000_000.0,
    )
    zero_rep = compute(zero, after_costs=True)
    taxed_rep = compute(taxed, after_costs=True)

    assert zero_rep.total_cost == 0.0
    assert zero_rep.net_pnl > 0.0
    assert taxed_rep.gross_pnl > 0.0
    assert taxed_rep.total_cost > 0.0
    assert taxed_rep.net_pnl < 0.0
    assert taxed_rep.gross_pnl > taxed_rep.net_pnl
    fills = taxed.equity.filter(pl.col("turnover") > 0)
    assert fills.height >= 10
    assert float(fills.get_column("turnover").mean()) > 0.9
