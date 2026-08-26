"""Purged k-fold and combinatorial purged CV.

Encoded from López de Prado, *Advances in Financial Machine Learning* (2018),
Chapter 7 (purging overlapping labels; embargo). Cited only because this
module implements those procedures, including combinatorial purged CV (CPCV).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import combinations


@dataclass(frozen=True)
class PurgedFold:
    train_dates: tuple[date, ...]
    test_dates: tuple[date, ...]
    purged_dates: tuple[date, ...]
    embargo_dates: tuple[date, ...]
    fold: int

    def validate(self) -> None:
        leak = set(self.train_dates) & set(self.test_dates)
        if leak:
            raise AssertionError(f"purged CV leak: {sorted(leak)[:5]}")
        if set(self.purged_dates) & set(self.train_dates):
            raise AssertionError("purged dates must be absent from train")
        if set(self.embargo_dates) & set(self.train_dates):
            raise AssertionError("embargo dates must be absent from train")


def _contiguous_groups(dates: list[date], n_groups: int) -> list[list[date]]:
    ordered = sorted(set(dates))
    n = len(ordered)
    if n_groups < 2 or n_groups > n:
        raise ValueError("n_groups must be in [2, n_dates]")
    sizes = [n // n_groups] * n_groups
    for i in range(n % n_groups):
        sizes[i] += 1
    out: list[list[date]] = []
    i = 0
    for s in sizes:
        out.append(ordered[i : i + s])
        i += s
    return out


def _purge_and_embargo(
    ordered: list[date],
    train: set[date],
    test: set[date],
    embargo_bars: int,
    label_horizon_bars: int,
) -> tuple[tuple[date, ...], tuple[date, ...], tuple[date, ...]]:
    """Drop train obs whose label interval overlaps test; then embargo after test.

    A label formed at index i uses bars (i, i + label_horizon_bars]. Overlap with
    test indices [t0, t1] iff i <= t1 and i + h >= t0. Embargo then drops the
    `embargo_bars` immediately after t1 (serial correlation into the next train).
    """
    idx = {d: i for i, d in enumerate(ordered)}
    test_idx = sorted(idx[d] for d in test if d in idx)
    if not test_idx:
        return tuple(sorted(train)), (), ()
    t0, t1 = test_idx[0], test_idx[-1]
    h = max(int(label_horizon_bars), 0)
    purged: set[date] = set()
    for d in list(train):
        i = idx[d]
        if i <= t1 and (i + h) >= t0:
            purged.add(d)
    embargo: set[date] = set()
    e = max(int(embargo_bars), 0)
    for j in range(t1 + 1, min(len(ordered), t1 + 1 + e)):
        embargo.add(ordered[j])
    train_kept = set(train) - test - purged - embargo
    return tuple(sorted(train_kept)), tuple(sorted(purged)), tuple(sorted(embargo))


def purged_kfold(
    dates: list[date],
    n_splits: int,
    embargo_bars: int = 0,
    label_horizon_bars: int = 1,
) -> list[PurgedFold]:
    groups = _contiguous_groups(dates, n_splits)
    ordered = sorted(set(dates))
    folds: list[PurgedFold] = []
    for i, test_g in enumerate(groups):
        test = set(test_g)
        train = set(ordered) - test
        kept, purged, embargo = _purge_and_embargo(
            ordered, train, test, embargo_bars, label_horizon_bars
        )
        fold = PurgedFold(kept, tuple(sorted(test)), purged, embargo, i)
        fold.validate()
        folds.append(fold)
    return folds


def combinatorial_purged_cv(
    dates: list[date],
    n_groups: int,
    n_test_groups: int,
    embargo_bars: int = 0,
    label_horizon_bars: int = 1,
) -> list[PurgedFold]:
    """CPCV: every combination of n_test_groups out of n_groups as the test set."""
    if n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("need 1 <= n_test_groups < n_groups")
    groups = _contiguous_groups(dates, n_groups)
    ordered = sorted(set(dates))
    folds: list[PurgedFold] = []
    for fold_i, test_ids in enumerate(combinations(range(n_groups), n_test_groups)):
        test: set[date] = set()
        for gid in test_ids:
            test.update(groups[gid])
        train = set(ordered) - test
        kept, purged, embargo = _purge_and_embargo(
            ordered, train, test, embargo_bars, label_horizon_bars
        )
        fold = PurgedFold(kept, tuple(sorted(test)), purged, embargo, fold_i)
        fold.validate()
        folds.append(fold)
    return folds
