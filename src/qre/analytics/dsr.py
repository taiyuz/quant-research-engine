"""Deflated Sharpe Ratio.

Implements Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality" (Journal of Portfolio
Management, 2014). That paper is cited in RESEARCH.md only because this module
encodes PSR / DSR; we do not cite work we did not implement.
"""
from __future__ import annotations

import math
from statistics import NormalDist

_Z = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sr: float,
    sr_benchmark: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true SR > sr_benchmark | observed sr). `sr` is NOT annualized."""
    if n_obs < 3 or not math.isfinite(sr) or not math.isfinite(sr_benchmark):
        return float("nan")
    denom_var = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if denom_var <= 0.0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom_var)
    return float(_Z.cdf(z))


def expected_max_sharpe(n_trials: int) -> float:
    """Expected max of n_trials independent standard-normal Sharpes (LdP approx)."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    a = _Z.inv_cdf(1.0 - 1.0 / n)
    b = _Z.inv_cdf(1.0 - 1.0 / (n * math.e))
    return (1.0 - _EULER_MASCHERONI) * a + _EULER_MASCHERONI * b


def deflated_sharpe(
    sr: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR = PSR evaluated at the expected-max Sharpe under n_trials tests.

    Pass the non-annualized Sharpe (mean/std of the period returns). A single
    pre-declared spec should use n_trials=1. If you searched 20 configs, pass 20.
    """
    sr0 = expected_max_sharpe(n_trials)
    return probabilistic_sharpe_ratio(sr, sr0, n_obs, skew=skew, kurtosis=kurtosis)


def non_annualized_sharpe(returns) -> float:
    import numpy as np

    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    s = float(x.std(ddof=1))
    if s == 0.0:
        return float("nan")
    return float(x.mean() / s)
