"""Look-ahead in labels: a target dated t with horizon h must not see past t+h.

Remaining-sample returns (close[T]/close[t]-1) are the sloppy construction.
They would fail an honest backtest the same way a centered z-score would.
No Sharpe is claimed. Fixture is labeled SYNTHETIC.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from qre.features.pipeline import assert_features_invariant_to_future
from qre.features.returns import add_simple_return
from qre.research.labels import (
    add_executable_forward_return,
    add_forward_return,
    assert_labels_invariant_past_horizon,
    label_horizon_end,
)
from qre.types import BarPanel, DataOrigin, PanelMetadata


def _panel() -> BarPanel:
    """SYNTHETIC close-only path with a late jump, so remaining-sample != short horizon."""
    start = date(2020, 1, 2)
    prices = [
        100.0,
        101.0,
        100.0,
        102.0,
        101.0,
        103.0,
        104.0,
        150.0,
        151.0,
        149.0,
        200.0,
        201.0,
    ]
    rows = []
    d = start
    for px in prices:
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
        d += timedelta(days=1)
    return BarPanel(
        frame=pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)),
        metadata=PanelMetadata(
            origin=DataOrigin.SYNTHETIC,
            notes="SYNTHETIC label-lookahead fixture — not a market",
        ),
    )


def _mutate_after(panel: BarPanel, after: date, factor: float = 3.0) -> BarPanel:
    mutated = panel.frame
    for col in ("open", "high", "low", "close"):
        mutated = mutated.with_columns(
            pl.when(pl.col("date") > after)
            .then(pl.col(col) * factor)
            .otherwise(pl.col(col))
            .alias(col)
        )
    return panel.copy_with_frame(mutated)


def _remaining_sample_return(panel: BarPanel) -> pl.DataFrame:
    """Sloppy: y_t = last_close / close_t - 1. Uses the end of the backtest."""
    last = panel.frame.group_by("symbol").agg(pl.col("close").last().alias("_last"))
    return (
        panel.frame.join(last, on="symbol")
        .with_columns((pl.col("_last") / pl.col("close") - 1.0).alias("y_remain"))
        .drop("_last")
    )


def _fwd(panel: BarPanel, horizon: int = 2) -> pl.DataFrame:
    return add_forward_return(panel.frame, horizon)


def test_forward_return_matches_shift() -> None:
    panel = _panel()
    horizon = 2
    frame = add_forward_return(panel.frame, horizon).sort("date")
    close = frame.get_column("close").to_list()
    y = frame.get_column("fwd_ret_2").to_list()
    for i in range(len(close) - horizon):
        expected = close[i + horizon] / close[i] - 1.0
        assert abs(y[i] - expected) < 1e-12
    assert all(v is None for v in y[-horizon:])


def test_forward_label_invariant_past_horizon() -> None:
    panel = _panel()
    cutoff = panel.dates[3]
    horizon = 2
    horizon_end = label_horizon_end(panel.dates, cutoff, horizon)
    mutated = _mutate_after(panel, horizon_end)
    assert_labels_invariant_past_horizon(
        lambda p: _fwd(p, horizon),
        panel,
        mutated,
        cutoff,
        horizon,
        ["fwd_ret_2"],
    )


def test_forward_label_depends_on_prices_inside_horizon() -> None:
    panel = _panel()
    cutoff = panel.dates[3]
    horizon = 2
    # Mutate the first bar after cutoff — inside the label window.
    mutated = _mutate_after(panel, cutoff)
    y0 = (
        add_forward_return(panel.frame, horizon)
        .filter(pl.col("date") == cutoff)
        .get_column("fwd_ret_2")
        .item()
    )
    y1 = (
        add_forward_return(mutated.frame, horizon)
        .filter(pl.col("date") == cutoff)
        .get_column("fwd_ret_2")
        .item()
    )
    assert y0 is not None and y1 is not None
    assert abs(y0 - y1) > 1e-9


def test_remaining_sample_label_leaks_past_horizon() -> None:
    """Sloppy backtest: label every bar with the return to the end of the sample.

    Mutating the last close changes y at an early cutoff. A finite-horizon
    label at the same cutoff does not. That is look-ahead in the target.
    """
    panel = _panel()
    cutoff = panel.dates[3]
    horizon = 2
    horizon_end = label_horizon_end(panel.dates, cutoff, horizon)
    assert panel.dates[-1] > horizon_end
    mutated = _mutate_after(panel, horizon_end)

    with pytest.raises(AssertionError, match="look-ahead leak in label"):
        assert_labels_invariant_past_horizon(
            _remaining_sample_return,
            panel,
            mutated,
            cutoff,
            horizon,
            ["y_remain"],
        )

    remain0 = (
        _remaining_sample_return(panel)
        .filter(pl.col("date") == cutoff)
        .get_column("y_remain")
        .item()
    )
    remain1 = (
        _remaining_sample_return(mutated)
        .filter(pl.col("date") == cutoff)
        .get_column("y_remain")
        .item()
    )
    assert abs(remain0 - remain1) > 1e-9

    assert_labels_invariant_past_horizon(
        lambda p: _fwd(p, horizon),
        panel,
        mutated,
        cutoff,
        horizon,
        ["fwd_ret_2"],
    )


def test_forward_return_fails_feature_invariance() -> None:
    """Putting y_t in the signal frame is look-ahead. Features at t must not."""
    panel = _panel()
    cutoff = panel.dates[4]
    mutated = _mutate_after(panel, cutoff)
    with pytest.raises(AssertionError, match="look-ahead leak"):
        assert_features_invariant_to_future(
            lambda p: add_forward_return(p.frame, 3),
            panel,
            mutated,
            cutoff,
            ["fwd_ret_3"],
        )


def test_executable_label_is_not_the_signal_bar_return() -> None:
    """fill_delay=1: the target starts at the fill, not at the signal close.

    A sloppy label y_t = close[t+1]/close[t]-1 is the untradeable gap
    between signal and fill (see test_next_bar_fill_does_not_earn_same_bar_move).
    """
    d0 = date(2020, 1, 2)
    d1 = date(2020, 1, 3)
    d2 = date(2020, 1, 6)
    rows = []
    for d, px in ((d0, 100.0), (d1, 110.0), (d2, 132.0)):
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
    panel = BarPanel(
        frame=pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)),
        metadata=PanelMetadata(
            origin=DataOrigin.SYNTHETIC,
            notes="SYNTHETIC executable-label fixture — not a market",
        ),
    )
    frame = add_simple_return(panel.frame)
    frame = add_forward_return(frame, 1)
    frame = add_executable_forward_return(frame, horizon=1, fill_delay=1)
    row0 = frame.filter(pl.col("date") == d0).row(0, named=True)
    row1 = frame.filter(pl.col("date") == d1).row(0, named=True)
    row2 = frame.filter(pl.col("date") == d2).row(0, named=True)
    # Untradeable gap (signal close -> fill close) is fwd_ret_1 at d0.
    assert abs(row0["fwd_ret_1"] - (110.0 / 100.0 - 1.0)) < 1e-12
    # First earned move (fill close -> next close) is exec label at d0,
    # equal to simple_return at d2, not at d1.
    assert abs(row0["exec_fwd_ret_1"] - (132.0 / 110.0 - 1.0)) < 1e-12
    assert abs(row2["simple_return"] - row0["exec_fwd_ret_1"]) < 1e-12
    assert abs(row1["simple_return"] - row0["exec_fwd_ret_1"]) > 1e-9
    assert row0["exec_fwd_ret_1"] != row0["fwd_ret_1"]


def test_horizon_and_fill_delay_rejected() -> None:
    panel = _panel()
    with pytest.raises(ValueError, match="horizon"):
        add_forward_return(panel.frame, 0)
    with pytest.raises(ValueError, match="fill_delay"):
        add_executable_forward_return(panel.frame, horizon=1, fill_delay=0)
