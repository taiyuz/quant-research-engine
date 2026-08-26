"""Walk-forward: test dates never appear in train; embargo is honored."""

from __future__ import annotations

from datetime import date, timedelta

from qre.research.walk_forward import expanding_splits, make_splits, rolling_splits


def _dates(n: int) -> list[date]:
    start = date(2020, 1, 2)
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_expanding_no_overlap() -> None:
    dates = _dates(400)
    splits = expanding_splits(dates, train_min_bars=100, test_bars=20, step_bars=20, embargo_bars=5)
    assert splits
    for s in splits:
        train = set(s.train_dates(dates))
        test = set(s.test_dates(dates))
        assert train.isdisjoint(test)
        assert s.train_end < s.test_start
        assert s.train_start == dates[0]


def test_rolling_no_overlap() -> None:
    dates = _dates(400)
    splits = rolling_splits(dates, train_bars=80, test_bars=20, step_bars=20, embargo_bars=3)
    assert splits
    for s in splits:
        train = set(s.train_dates(dates))
        test = set(s.test_dates(dates))
        assert train.isdisjoint(test)
        assert len(s.train_dates(dates)) == 80
        assert len(s.test_dates(dates)) == 20


def test_embargo_gap() -> None:
    dates = _dates(200)
    embargo = 7
    splits = expanding_splits(
        dates, train_min_bars=50, test_bars=10, step_bars=10, embargo_bars=embargo
    )
    for s in splits:
        t_end = dates.index(s.train_end)
        t_start = dates.index(s.test_start)
        gap = t_start - t_end - 1
        assert gap >= embargo
        embargoed = set(dates[t_end + 1 : t_start])
        assert embargoed.isdisjoint(set(s.train_dates(dates)))
        assert embargoed.isdisjoint(set(s.test_dates(dates)))


def test_make_splits_dispatch() -> None:
    dates = _dates(150)
    a = make_splits(dates, "expanding", train_bars=60, test_bars=10, step_bars=10, embargo_bars=0)
    b = make_splits(dates, "rolling", train_bars=60, test_bars=10, step_bars=10, embargo_bars=0)
    assert a and b
    assert a[0].train_start == dates[0]
    assert b[0].train_start == dates[0]
