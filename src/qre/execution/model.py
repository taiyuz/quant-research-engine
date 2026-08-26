from __future__ import annotations

import numpy as np
import polars as pl

from qre.types import CostModel

ExecutionModel = CostModel


def apply_costs(traded_notional: pl.Series | np.ndarray | list[float], costs: CostModel) -> np.ndarray:
    x = np.asarray(traded_notional, dtype=float)
    return np.abs(x) * costs.cost_rate
