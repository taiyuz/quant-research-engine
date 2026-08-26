"""Performance analytics. Always carry data_origin and after_costs.

Annualization is 252 trading days. Sharpe is mean/std * sqrt(252); zero
vol returns NaN, not inf. Sortino uses downside deviation vs MAR=0.
Max drawdown is computed on the NAV equity curve.

These numbers on SYNTHETIC panels are a pipeline demo. They are not alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import nan, sqrt

import numpy as np
import polars as pl

from qre.portfolio.simulator import SimulationResult
from qre.types import TRADING_DAYS_PER_YEAR, DataOrigin


@dataclass(frozen=True)
class PerformanceReport:
    data_origin: DataOrigin
    after_costs: bool
    n_bars: int
    total_return: float
    pnl: float
    vol_annual: float
    sharpe: float
    sortino: float
    max_drawdown: float
    mean_turnover: float
    total_cost: float
    mean_gross_exposure: float
    mean_net_exposure: float
    gross_pnl: float
    net_pnl: float

    def as_dict(self) -> dict[str, object]:
        return {
            "data_origin": self.data_origin.value,
            "after_costs": self.after_costs,
            "n_bars": self.n_bars,
            "total_return": self.total_return,
            "pnl": self.pnl,
            "vol_annual": self.vol_annual,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "mean_turnover": self.mean_turnover,
            "total_cost": self.total_cost,
            "mean_gross_exposure": self.mean_gross_exposure,
            "mean_net_exposure": self.mean_net_exposure,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
        }


def _ann_factor() -> float:
    return sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: np.ndarray) -> float:
    if returns.size == 0:
        return nan
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if sd == 0.0:
        return nan
    return mu / sd * _ann_factor()


def sortino_ratio(returns: np.ndarray, mar: float = 0.0) -> float:
    """Downside deviation = sqrt(mean(min(r - MAR, 0)^2))."""
    if returns.size == 0:
        return nan
    mu = float(np.mean(returns))
    downside = np.minimum(returns - mar, 0.0)
    dd = float(np.sqrt(np.mean(downside * downside)))
    if dd == 0.0:
        return nan
    return mu / dd * _ann_factor()


def max_drawdown(equity: np.ndarray) -> float:
    """Peak-to-trough on the equity curve; returned as a negative fraction."""
    if equity.size == 0:
        return nan
    peak = np.maximum.accumulate(equity)
    dd = equity / np.where(peak == 0.0, np.nan, peak) - 1.0
    return float(np.nanmin(dd))


def annualized_vol(returns: np.ndarray) -> float:
    if returns.size <= 1:
        return nan
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return nan
    return sd * _ann_factor()


def compute(result: SimulationResult, after_costs: bool = True) -> PerformanceReport:
    eq = result.equity.sort("date")
    col = "net_ret" if after_costs else "gross_ret"
    rets = eq.get_column(col).to_numpy().astype(float)
    if rets.size > 1:
        stat_rets = rets[1:]
    else:
        stat_rets = rets
    nav = eq.get_column("nav").to_numpy().astype(float)
    initial = result.initial_nav
    final = float(nav[-1]) if nav.size else initial
    total_return = final / initial - 1.0 if initial else nan
    pnl = final - initial
    costs = eq.get_column("cost").to_numpy().astype(float)
    total_cost = float(np.sum(costs))
    gross_rets = eq.get_column("gross_ret").to_numpy().astype(float)
    if gross_rets.size > 1:
        gnav = initial * np.cumprod(1.0 + gross_rets)
        gross_pnl = float(gnav[-1] - initial)
    else:
        gross_pnl = 0.0
    return PerformanceReport(
        data_origin=result.data_origin,
        after_costs=after_costs,
        n_bars=int(eq.height),
        total_return=total_return,
        pnl=pnl,
        vol_annual=annualized_vol(stat_rets),
        sharpe=sharpe_ratio(stat_rets),
        sortino=sortino_ratio(stat_rets),
        max_drawdown=max_drawdown(nav),
        mean_turnover=float(np.mean(eq.get_column("turnover").to_numpy().astype(float))),
        total_cost=total_cost,
        mean_gross_exposure=float(np.mean(eq.get_column("gross_exposure").to_numpy().astype(float))),
        mean_net_exposure=float(np.mean(eq.get_column("net_exposure").to_numpy().astype(float))),
        gross_pnl=gross_pnl,
        net_pnl=pnl,
    )
