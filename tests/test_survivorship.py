"""Delisted names must not appear after delist_date. PIT membership as-of date."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from qre.data.loader import generate_synthetic_ohlcv
from qre.data.universe import MembershipRecord, PointInTimeUniverse
from qre.types import BarPanel, DataOrigin, PanelMetadata


def test_synthetic_delist_drops_post_delist_bars() -> None:
    panel, universe = generate_synthetic_ohlcv(
        n_symbols=5, n_bars=80, seed=8, process="gbm", delist_mid_sample=True
    )
    rec = [r for r in universe.records if r.delist_date is not None]
    assert rec, "generator should delist at least one name"
    dead = rec[0]
    post = panel.frame.filter(
        (pl.col("symbol") == dead.symbol) & (pl.col("date") > dead.delist_date)
    )
    assert post.height == 0
    universe.assert_no_post_delist_bars(panel)
    on_delist = panel.frame.filter(
        (pl.col("symbol") == dead.symbol) & (pl.col("date") == dead.delist_date)
    )
    assert on_delist.height == 1
    later = [d for d in panel.dates if d > dead.delist_date]
    if later:
        assert dead.symbol not in universe.members_asof(later[0])
    _ = timedelta


def test_filter_panel_enforces_membership() -> None:
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    rows = []
    for d in dates:
        for sym, px in (("AAA", 10.0), ("BBB", 20.0)):
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "volume": 1,
                }
            )
    panel = BarPanel(
        frame=pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)),
        metadata=PanelMetadata(origin=DataOrigin.SYNTHETIC, notes="SYNTHETIC fixture"),
    )
    uni = PointInTimeUniverse(
        records=(
            MembershipRecord("AAA", dates[0], None),
            MembershipRecord("BBB", dates[0], dates[1]),
        )
    )
    filtered = uni.filter_panel(panel)
    bbb = filtered.frame.filter(pl.col("symbol") == "BBB").get_column("date").to_list()
    assert dates[2] not in bbb
    assert dates[1] in bbb
    assert "BBB" not in uni.members_asof(dates[2])
    assert "AAA" in uni.members_asof(dates[2])


def test_members_asof_is_point_in_time() -> None:
    uni = PointInTimeUniverse(
        records=(
            MembershipRecord("OLD", date(2019, 1, 1), date(2020, 6, 1)),
            MembershipRecord("NEW", date(2020, 3, 1), None),
        )
    )
    assert uni.members_asof(date(2019, 6, 1)) == ["OLD"]
    assert set(uni.members_asof(date(2020, 4, 1))) == {"OLD", "NEW"}
    assert uni.members_asof(date(2020, 6, 2)) == ["NEW"]
