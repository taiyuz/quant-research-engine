from __future__ import annotations

from datetime import date, timedelta
from math import comb

from qre.research.purged_cv import combinatorial_purged_cv, purged_kfold


def _dates(n: int) -> list[date]:
    start = date(2020, 1, 2)
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_purged_kfold_no_train_test_overlap() -> None:
    dates = _dates(120)
    folds = purged_kfold(dates, n_splits=5, embargo_bars=3, label_horizon_bars=2)
    assert len(folds) == 5
    for f in folds:
        f.validate()
        assert set(f.train_dates).isdisjoint(set(f.test_dates))
        # label horizon 2 purges the 2 train bars immediately before test
        test_idx = [dates.index(d) for d in f.test_dates]
        t0 = min(test_idx)
        for d in f.train_dates:
            i = dates.index(d)
            assert not (i <= max(test_idx) and i + 2 >= t0)


def test_embargo_removed_from_train() -> None:
    dates = _dates(90)
    folds = purged_kfold(dates, n_splits=3, embargo_bars=4, label_horizon_bars=1)
    for f in folds:
        t1 = dates.index(max(f.test_dates))
        after = dates[t1 + 1 : t1 + 1 + 4]
        assert set(after).isdisjoint(set(f.train_dates))
        assert set(after) <= set(f.embargo_dates) | set(dates[t1 + 1 :])


def test_cpcv_fold_count() -> None:
    dates = _dates(60)
    folds = combinatorial_purged_cv(
        dates, n_groups=6, n_test_groups=2, embargo_bars=2, label_horizon_bars=1
    )
    assert len(folds) == comb(6, 2)
    for f in folds:
        f.validate()
        assert set(f.train_dates).isdisjoint(set(f.test_dates))


def test_purge_gap_before_test_never_enters_train() -> None:
    """Bars whose labels overlap the test window form a purge gap and stay out of train.

    AFML Ch.7: a label formed at index i with horizon h overlaps a test window
    starting at t0 when i <= t1 and i + h >= t0. With embargo_bars=0, the h
    sessions immediately before an interior test fold must be in purged_dates,
    not train.
    """
    dates = _dates(80)
    horizon = 4
    folds = purged_kfold(dates, n_splits=4, embargo_bars=0, label_horizon_bars=horizon)
    interior = [f for f in folds if dates.index(min(f.test_dates)) >= horizon]
    assert interior
    for f in interior:
        t0 = dates.index(min(f.test_dates))
        gap = dates[t0 - horizon : t0]
        assert len(gap) == horizon
        for d in gap:
            assert d not in f.train_dates
            assert d not in f.test_dates
            assert d in f.purged_dates
        assert set(gap) <= set(f.purged_dates)
