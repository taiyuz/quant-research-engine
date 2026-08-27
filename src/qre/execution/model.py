"""Fee-plus-slippage execution with next-bar fill.

Default convention
------------------
Signal / target weights dated t are a function of close[t] (and history
<= t). They are filled at the close of t+1. You cannot trade the same
bar's close you used to form the signal.

Cost
----
cost = (commission_bps + slippage_bps) / 1e4 * abs(delta_weight) * nav

That is a proportional cost on traded notional. Gross PnL >= net PnL
whenever turnover > 0. The simulator records both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ExecutionModel:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    fill_delay: int = 1

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must be >= 0")
        if self.fill_delay < 1:
            raise ValueError(
                "fill_delay must be >= 1 so the same-bar close is not tradable"
            )

    @property
    def cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4

    def cost_on_notional(self, traded_notional: float) -> float:
        return self.cost_rate * abs(traded_notional)

    def cost_from_weight_delta(self, abs_delta_weight: float, nav: float) -> float:
        """cost = (commission_bps + slippage_bps)/1e4 * abs(delta_weight) * nav"""
        return self.cost_rate * abs(abs_delta_weight) * nav


def apply_costs(
    traded_notional: pl.Series | np.ndarray | list[float], costs: ExecutionModel
) -> np.ndarray:
    x = np.asarray(traded_notional, dtype=float)
    return np.abs(x) * costs.cost_rate
