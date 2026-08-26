from __future__ import annotations

import math

from qre.analytics.dsr import (
    deflated_sharpe,
    expected_max_sharpe,
    non_annualized_sharpe,
    probabilistic_sharpe_ratio,
)


def test_expected_max_sharpe_increases_with_trials() -> None:
    assert expected_max_sharpe(1) == 0.0
    assert expected_max_sharpe(5) > expected_max_sharpe(2)
    assert expected_max_sharpe(50) > expected_max_sharpe(5)


def test_psr_one_sided() -> None:
    # large positive SR vs 0 with many obs -> PSR near 1
    psr = probabilistic_sharpe_ratio(sr=0.2, sr_benchmark=0.0, n_obs=500)
    assert psr > 0.99
    # SR below benchmark -> PSR < 0.5
    low = probabilistic_sharpe_ratio(sr=-0.05, sr_benchmark=0.0, n_obs=200)
    assert low < 0.5


def test_dsr_penalizes_multiple_testing() -> None:
    sr = non_annualized_sharpe([0.001, 0.002, -0.001, 0.0015] * 80)
    dsr_one = deflated_sharpe(sr, n_obs=320, n_trials=1)
    dsr_many = deflated_sharpe(sr, n_obs=320, n_trials=50)
    assert math.isfinite(dsr_one) and math.isfinite(dsr_many)
    assert dsr_many < dsr_one


def test_zero_vol_nan() -> None:
    assert math.isnan(non_annualized_sharpe([0.0, 0.0, 0.0]))
