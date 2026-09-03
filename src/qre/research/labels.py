"""Finite-horizon forward-return labels. These are targets, not features.

A label dated t with horizon h is

    y_t = close[t+h] / close[t] - 1

It uses future prices by construction. That is allowed for a *target*.
It is not allowed for a feature or a signal. A feature row dated t must
be invariant to prices after t (see qre.features.pipeline). A label row
dated t must be invariant to prices after t+h, and must move if prices
inside (t, t+h] move.

LOOK-AHEAD IN LABELS
--------------------
Sloppy backtests date an unbounded target at t: remaining return to the
end of the sample, "was this the peak", a barrier that never expires.
Those y_t values change when you edit the last bar of the backtest.
That is the same class of bug as a centered z-score, just on the target.
Purged CV (AFML Ch.7) exists because even a *finite* horizon label
overlaps the test window; an infinite horizon cannot be purged.

The executable target, given fill_delay >= 1, starts at the fill bar:

    y_t = close[t+fill_delay+h] / close[t+fill_delay] - 1

The same-bar simple return is the move you cannot trade (signal at t
uses close[t]; fill is t+1). Do not use it as the label either.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from qre.types import BarPanel


def add_forward_return(
    frame: pl.DataFrame,
    horizon: int,
    price_col: str = "close",
    alias: str | None = None,
) -> pl.DataFrame:
    """y_t = close[t+horizon] / close[t] - 1. Null on the last `horizon` bars.

    This column is a LABEL. Do not pass it to a strategy as a signal.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    name = alias or f"fwd_ret_{horizon}"
    frame = frame.sort(["symbol", "date"])
    return frame.with_columns(
        (pl.col(price_col).shift(-horizon).over("symbol") / pl.col(price_col) - 1.0).alias(
            name
        )
    )


def add_executable_forward_return(
    frame: pl.DataFrame,
    horizon: int,
    fill_delay: int = 1,
    price_col: str = "close",
    alias: str | None = None,
) -> pl.DataFrame:
    """Return earned if filled at t+fill_delay and held `horizon` bars.

    y_t = close[t+fill_delay+horizon] / close[t+fill_delay] - 1

    Known only at t+fill_delay+horizon. fill_delay must be >= 1 so the
    signal bar is not the entry. The same-bar simple return is not this.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if fill_delay < 1:
        raise ValueError("fill_delay must be >= 1 so the signal bar is not the entry")
    name = alias or f"exec_fwd_ret_{horizon}"
    frame = frame.sort(["symbol", "date"])
    entry = pl.col(price_col).shift(-fill_delay).over("symbol")
    exit_px = pl.col(price_col).shift(-(fill_delay + horizon)).over("symbol")
    return frame.with_columns((exit_px / entry - 1.0).alias(name))


def label_horizon_end(dates: list[date], cutoff: date, horizon: int) -> date:
    """Event-time date of cutoff + horizon bars (not calendar days)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    ordered = sorted(dates)
    try:
        i = ordered.index(cutoff)
    except ValueError as exc:
        raise ValueError(f"cutoff {cutoff} is not in dates") from exc
    j = i + horizon
    if j >= len(ordered):
        raise ValueError("cutoff + horizon is off the end of the panel")
    return ordered[j]


def assert_labels_invariant_past_horizon(
    label_fn,
    panel: BarPanel,
    panel_mutated_after_horizon: BarPanel,
    cutoff: date,
    horizon: int,
    label_cols: list[str],
) -> None:
    """Labels dated t must not depend on prices after t+horizon.

    `panel` and `panel_mutated_after_horizon` must agree through
    cutoff+horizon (inclusive) and differ after. label_fn(panel) rows
    with date <= cutoff must match.
    """
    horizon_end = label_horizon_end(panel.dates, cutoff, horizon)
    thru_cols = ["date", "symbol", "close"]
    a_thru = (
        panel.frame.filter(pl.col("date") <= horizon_end)
        .select(thru_cols)
        .sort(["symbol", "date"])
    )
    b_thru = (
        panel_mutated_after_horizon.frame.filter(pl.col("date") <= horizon_end)
        .select(thru_cols)
        .sort(["symbol", "date"])
    )
    if a_thru.shape != b_thru.shape or not a_thru.equals(b_thru):
        raise AssertionError(
            "panels must agree through cutoff+horizon; mutate only after that"
        )
    a_after = panel.frame.filter(pl.col("date") > horizon_end).get_column("close")
    b_after = panel_mutated_after_horizon.frame.filter(
        pl.col("date") > horizon_end
    ).get_column("close")
    if a_after.len() == 0 or b_after.len() == 0:
        raise AssertionError("need bars after cutoff+horizon to test label look-ahead")
    if not bool((a_after != b_after).any()):
        raise AssertionError("mutated panel must differ after cutoff+horizon")

    lab_a = label_fn(panel)
    lab_b = label_fn(panel_mutated_after_horizon)
    cols = ["date", "symbol", *label_cols]
    a = lab_a.filter(pl.col("date") <= cutoff).select(cols).sort(["symbol", "date"])
    b = lab_b.filter(pl.col("date") <= cutoff).select(cols).sort(["symbol", "date"])
    if a.shape != b.shape:
        raise AssertionError(f"label look-ahead shape mismatch: {a.shape} vs {b.shape}")
    joined = a.join(b, on=["date", "symbol"], how="inner", suffix="_mut")
    for col in label_cols:
        left = joined.get_column(col)
        right = joined.get_column(f"{col}_mut")
        both_null = left.is_null() & right.is_null()
        close = (left - right).abs() <= 1e-12
        ok = both_null | close
        if not bool(ok.all()):
            n_bad = int((~ok).sum())
            raise AssertionError(
                f"look-ahead leak in label {col}: {n_bad} rows through {cutoff} "
                f"changed after mutating prices after {horizon_end} "
                f"(declared horizon={horizon})"
            )
