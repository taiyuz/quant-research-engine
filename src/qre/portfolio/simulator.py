"""Portfolio simulator: apply next-bar fills and execution costs to target weights.

Loop (for dates d_0 ... d_{T-1}):
1. From d_{i-1} close to d_i close, held weights earn simple returns.
   Dollar weights drift with prices.
2. If i >= fill_delay, fill the target that was formed at d_{i-fill_delay}.
   Cost is charged on sum(|target - drifted|) * nav.
3. Names missing from today's bar (delisted) are flattened at the previous
   close: zero return today, remaining weight is traded to 0 and paid as cost.

Outputs an equity curve plus per-bar turnover, costs, and exposures.
data_origin is copied from the price panel so analytics cannot forget it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from qre.execution.model import ExecutionModel
from qre.types import BarPanel, DataOrigin


@dataclass
class SimulationResult:
    equity: pl.DataFrame
    holdings: pl.DataFrame
    data_origin: DataOrigin
    after_costs: bool
    initial_nav: float
    fill_delay: int
    notes: str = ""
    extra: dict[str, float] = field(default_factory=dict)


def simulate(
    panel: BarPanel,
    weights: pl.DataFrame,
    execution: ExecutionModel,
    initial_nav: float = 1_000_000.0,
) -> SimulationResult:
    if initial_nav <= 0:
        raise ValueError("initial_nav must be positive")
    required = {"date", "symbol", "target_weight"}
    if not required.issubset(set(weights.columns)):
        raise ValueError(f"weights missing {required - set(weights.columns)}")

    prices = {
        (r["date"], r["symbol"]): float(r["close"])
        for r in panel.frame.select("date", "symbol", "close").iter_rows(named=True)
    }
    dates: list[date] = sorted(panel.frame.get_column("date").unique().to_list())
    symbols_on: dict[date, set[str]] = {}
    for r in panel.frame.select("date", "symbol").iter_rows(named=True):
        symbols_on.setdefault(r["date"], set()).add(r["symbol"])

    target_on: dict[date, dict[str, float]] = {}
    for r in weights.select("date", "symbol", "target_weight").iter_rows(named=True):
        target_on.setdefault(r["date"], {})[r["symbol"]] = float(r["target_weight"])

    held: dict[str, float] = {}
    nav = initial_nav
    delay = execution.fill_delay

    equity_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []

    for i, d in enumerate(dates):
        gross_ret = 0.0
        cost = 0.0
        turnover = 0.0
        nav_before = nav

        if i > 0:
            prev = dates[i - 1]
            drifted: dict[str, float] = {}
            dollar_pnl = 0.0
            for sym, w in held.items():
                p0 = prices.get((prev, sym))
                p1 = prices.get((d, sym))
                if p0 is None or p0 <= 0:
                    r = 0.0
                elif p1 is None:
                    r = 0.0
                else:
                    r = p1 / p0 - 1.0
                dollar_pnl += w * nav_before * r
                if p1 is None:
                    drifted[sym] = 0.0
                else:
                    drifted[sym] = w * (1.0 + r)
            nav_after_move = nav_before + dollar_pnl
            if nav_after_move != 0:
                drifted = {s: v * nav_before / nav_after_move for s, v in drifted.items()}
            else:
                drifted = {s: 0.0 for s in drifted}
            gross_ret = (nav_after_move / nav_before - 1.0) if nav_before != 0 else 0.0
            nav = nav_after_move

            if i >= delay:
                signal_date = dates[i - delay]
                target = dict(target_on.get(signal_date, {}))
                live = symbols_on.get(d, set())
                for s in list(target):
                    if s not in live:
                        target[s] = 0.0
                names = set(drifted) | set(target) | live
                abs_delta = 0.0
                new_held: dict[str, float] = {}
                for s in names:
                    tw = target.get(s, 0.0)
                    if s not in live:
                        tw = 0.0
                    dw = tw - drifted.get(s, 0.0)
                    abs_delta += abs(dw)
                    if s in live and tw != 0.0:
                        new_held[s] = tw
                cost = execution.cost_from_weight_delta(abs_delta, nav)
                turnover = abs_delta
                nav = nav - cost
                held = new_held
            else:
                held = {s: w for s, w in drifted.items() if s in symbols_on.get(d, set())}
        else:
            held = {}

        net_ret = (nav / nav_before - 1.0) if i > 0 and nav_before != 0 else 0.0
        gross_exp = sum(abs(w) for w in held.values())
        net_exp = sum(held.values())
        equity_rows.append(
            {
                "date": d,
                "nav": nav,
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "cost": cost,
                "turnover": turnover,
                "gross_exposure": gross_exp,
                "net_exposure": net_exp,
            }
        )
        for s, w in sorted(held.items()):
            holding_rows.append({"date": d, "symbol": s, "weight": w})

    equity = pl.DataFrame(equity_rows).with_columns(pl.col("date").cast(pl.Date))
    holdings = (
        pl.DataFrame(holding_rows).with_columns(pl.col("date").cast(pl.Date))
        if holding_rows
        else pl.DataFrame({"date": [], "symbol": [], "weight": []})
    )
    return SimulationResult(
        equity=equity,
        holdings=holdings,
        data_origin=panel.origin,
        after_costs=True,
        initial_nav=initial_nav,
        fill_delay=delay,
        notes=(
            "Signal at t filled at t+fill_delay close. "
            f"origin={panel.origin.value}. Costs subtracted from NAV."
        ),
    )
