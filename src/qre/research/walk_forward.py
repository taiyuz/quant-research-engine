"""Walk-forward / train-test splits with an optional embargo.

Test window dates never appear in train. Parameter choice, if any, belongs
on train only. An embargo of k bars leaves a gap between train_end and
test_start so that overlapping rolling features cannot leak.

WALK-FORWARD IS NOT A PANACEA
-----------------------------
If you run twenty configs, pick the one with the best walk-forward Sharpe,
and then report that Sharpe as out-of-sample, you have used the test path
to select the spec. That is still multiple testing. See RESEARCH.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    embargo_bars: int
    fold: int

    def train_dates(self, dates: list[date]) -> list[date]:
        return [d for d in dates if self.train_start <= d <= self.train_end]

    def test_dates(self, dates: list[date]) -> list[date]:
        return [d for d in dates if self.test_start <= d <= self.test_end]


def expanding_splits(
    dates: list[date],
    train_min_bars: int,
    test_bars: int,
    step_bars: int,
    embargo_bars: int = 0,
) -> list[WalkForwardSplit]:
    """Expanding train window, rolling test window, optional embargo."""
    if train_min_bars < 1 or test_bars < 1 or step_bars < 1:
        raise ValueError("train_min_bars, test_bars, step_bars must be >= 1")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")
    ordered = sorted(dates)
    n = len(ordered)
    out: list[WalkForwardSplit] = []
    fold = 0
    train_end_idx = train_min_bars - 1
    while True:
        test_start_idx = train_end_idx + 1 + embargo_bars
        test_end_idx = test_start_idx + test_bars - 1
        if test_end_idx >= n:
            break
        split = WalkForwardSplit(
            train_start=ordered[0],
            train_end=ordered[train_end_idx],
            test_start=ordered[test_start_idx],
            test_end=ordered[test_end_idx],
            embargo_bars=embargo_bars,
            fold=fold,
        )
        _validate_no_leak(split, ordered)
        out.append(split)
        fold += 1
        train_end_idx += step_bars
    return out


def rolling_splits(
    dates: list[date],
    train_bars: int,
    test_bars: int,
    step_bars: int,
    embargo_bars: int = 0,
) -> list[WalkForwardSplit]:
    """Rolling (fixed-length) train window, rolling test window, optional embargo."""
    if train_bars < 1 or test_bars < 1 or step_bars < 1:
        raise ValueError("train_bars, test_bars, step_bars must be >= 1")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")
    ordered = sorted(dates)
    n = len(ordered)
    out: list[WalkForwardSplit] = []
    fold = 0
    train_start_idx = 0
    while True:
        train_end_idx = train_start_idx + train_bars - 1
        test_start_idx = train_end_idx + 1 + embargo_bars
        test_end_idx = test_start_idx + test_bars - 1
        if test_end_idx >= n:
            break
        split = WalkForwardSplit(
            train_start=ordered[train_start_idx],
            train_end=ordered[train_end_idx],
            test_start=ordered[test_start_idx],
            test_end=ordered[test_end_idx],
            embargo_bars=embargo_bars,
            fold=fold,
        )
        _validate_no_leak(split, ordered)
        out.append(split)
        fold += 1
        train_start_idx += step_bars
    return out


def _validate_no_leak(split: WalkForwardSplit, dates: list[date]) -> None:
    train = set(split.train_dates(dates))
    test = set(split.test_dates(dates))
    overlap = train & test
    if overlap:
        raise AssertionError(f"walk-forward leak: {len(overlap)} dates in train and test")
    if split.train_end >= split.test_start:
        raise AssertionError("train_end must be strictly before test_start")
    if split.embargo_bars:
        t_end = dates.index(split.train_end)
        t_start = dates.index(split.test_start)
        gap = t_start - t_end - 1
        if gap < split.embargo_bars:
            raise AssertionError(
                f"embargo too small: gap={gap} bars, required={split.embargo_bars}"
            )


def make_splits(
    dates: list[date],
    mode: str,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    embargo_bars: int = 0,
) -> list[WalkForwardSplit]:
    mode_n = mode.strip().lower()
    if mode_n == "expanding":
        return expanding_splits(dates, train_bars, test_bars, step_bars, embargo_bars)
    if mode_n == "rolling":
        return rolling_splits(dates, train_bars, test_bars, step_bars, embargo_bars)
    raise ValueError(f"unknown walk-forward mode {mode!r}")
