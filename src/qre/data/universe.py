"""Point-in-time universe membership and survivorship handling.

SURVIVORSHIP BIAS
-----------------
A live research universe is a function of the as-of date: who was a member
on that date, not who is a member today. Building a panel from today's
survivors (the names that never delisted, never went to zero, never left
the index) inflates historical returns of almost every long-biased idea.

This module stores membership as (symbol, start_date, delist_date) records.
members_asof(d) returns names with start_date <= d <= delist_date.
filter_panel drops any bar whose (date, symbol) is not a member that day.

The synthetic generator may delist a symbol mid-sample. After delist_date
that symbol MUST NOT appear in the panel. Tests enforce this; do not
repair a missing name by forward-filling it past delist.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from qre.types import BarPanel


@dataclass(frozen=True)
class MembershipRecord:
    """Inclusive membership interval for one name.

    delist_date is the last date the name is tradable. None means the name
    remains listed through the end of the sample (not listed forever in
    production — just no delist event in this panel).
    """

    symbol: str
    start_date: date
    delist_date: date | None = None

    def is_member(self, asof: date) -> bool:
        if asof < self.start_date:
            return False
        if self.delist_date is None:
            return True
        return asof <= self.delist_date


@dataclass(frozen=True)
class PointInTimeUniverse:
    records: tuple[MembershipRecord, ...]

    def members_asof(self, asof: date) -> list[str]:
        return [r.symbol for r in self.records if r.is_member(asof)]

    def delist_date_of(self, symbol: str) -> date | None:
        for r in self.records:
            if r.symbol == symbol:
                return r.delist_date
        raise KeyError(symbol)

    def filter_panel(self, panel: BarPanel) -> BarPanel:
        """Drop bars that violate point-in-time membership.

        A delisted name must not appear after delist_date. This is the
        mechanical enforcement of the survivorship note above.
        """
        rec_rows = [
            {
                "symbol": r.symbol,
                "start_date": r.start_date,
                "delist_date": r.delist_date,
            }
            for r in self.records
        ]
        membership = pl.DataFrame(rec_rows).with_columns(
            pl.col("start_date").cast(pl.Date),
            pl.col("delist_date").cast(pl.Date),
        )
        frame = panel.frame.join(membership, on="symbol", how="inner")
        listed = pl.col("date") >= pl.col("start_date")
        not_yet_delisted = pl.col("delist_date").is_null() | (
            pl.col("date") <= pl.col("delist_date")
        )
        frame = (
            frame.filter(listed & not_yet_delisted)
            .drop(["start_date", "delist_date"])
            .sort(["symbol", "date"])
        )
        return panel.copy_with_frame(frame)

    def assert_no_post_delist_bars(self, panel: BarPanel) -> None:
        for rec in self.records:
            if rec.delist_date is None:
                continue
            post = panel.frame.filter(
                (pl.col("symbol") == rec.symbol)
                & (pl.col("date") > rec.delist_date)
            )
            if post.height > 0:
                raise AssertionError(
                    f"survivorship violation: {rec.symbol} has "
                    f"{post.height} bars after delist_date {rec.delist_date}"
                )


def universe_from_panel(panel: BarPanel) -> PointInTimeUniverse:
    """Infer membership from whatever bars are present (last bar = delist if short)."""
    dates = panel.dates
    if not dates:
        return PointInTimeUniverse(records=())
    sample_end = dates[-1]
    records: list[MembershipRecord] = []
    for symbol in panel.symbols:
        sub = panel.frame.filter(pl.col("symbol") == symbol).get_column("date")
        start = sub.min()
        end = sub.max()
        delist = None if end == sample_end else end
        records.append(MembershipRecord(symbol=symbol, start_date=start, delist_date=delist))
    return PointInTimeUniverse(records=tuple(records))
