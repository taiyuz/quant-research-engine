"""CLI: qre-backtest <config.yaml>

Runs the research pipeline on labeled synthetic data (or a user csv).
Prints the SYNTHETIC banner whenever origin is SYNTHETIC. Sample metrics
are a pipeline demo, not an edge.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from qre.analytics.metrics import PerformanceReport, compute
from qre.data.loader import generate_synthetic_ohlcv, load_ohlcv_csv, load_ohlcv_parquet
from qre.execution.model import ExecutionModel
from qre.features.pipeline import build_features
from qre.portfolio.simulator import simulate
from qre.research.walk_forward import make_splits
from qre.strategies.base import build_strategy
from qre.types import (
    SYNTHETIC_BANNER,
    SYNTHETIC_FOOTER,
    BarPanel,
    DataOrigin,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    return cfg


def panel_from_config(cfg: dict[str, Any]) -> BarPanel:
    data = cfg.get("data") or {}
    src = data.get("path")
    if src:
        p = Path(src)
        if p.suffix.lower() == ".parquet":
            return load_ohlcv_parquet(p)
        return load_ohlcv_csv(p)
    process = str(data.get("process", "gbm"))
    panel, universe = generate_synthetic_ohlcv(
        n_symbols=int(data.get("n_symbols", 8)),
        n_bars=int(data.get("n_bars", 504)),
        seed=int(data.get("seed", 42)),
        process=process,
        delist_mid_sample=bool(data.get("delist_mid_sample", False)),
    )
    panel = universe.filter_panel(panel)
    return panel


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.6f}"


def format_report(report: PerformanceReport, title: str) -> str:
    lines = [
        title,
        f"  data_origin:           {report.data_origin.value}",
        f"  after_costs:           {report.after_costs}",
        f"  n_bars:                {report.n_bars}",
        f"  total_return:          {_fmt(report.total_return)}",
        f"  pnl:                   {_fmt(report.pnl)}",
        f"  gross_pnl:             {_fmt(report.gross_pnl)}",
        f"  net_pnl:               {_fmt(report.net_pnl)}",
        f"  vol_annual:            {_fmt(report.vol_annual)}",
        f"  sharpe:                {_fmt(report.sharpe)}",
        f"  sortino:               {_fmt(report.sortino)}",
        f"  max_drawdown:          {_fmt(report.max_drawdown)}",
        f"  mean_turnover:         {_fmt(report.mean_turnover)}",
        f"  total_cost:            {_fmt(report.total_cost)}",
        f"  mean_gross_exposure:   {_fmt(report.mean_gross_exposure)}",
        f"  mean_net_exposure:     {_fmt(report.mean_net_exposure)}",
    ]
    return "\n".join(lines)


def run_backtest(cfg: dict[str, Any]) -> tuple[PerformanceReport, list[str]]:
    logs: list[str] = []
    panel = panel_from_config(cfg)
    lookback = int(cfg.get("lookback", 20))
    strategy = build_strategy(
        str(cfg.get("strategy", "momentum")),
        lookback=lookback,
        n_long=cfg.get("n_long"),
        leg_a=cfg.get("leg_a"),
        leg_b=cfg.get("leg_b"),
    )
    costs = cfg.get("costs") or {}
    execution = ExecutionModel(
        commission_bps=float(costs.get("commission_bps", 1.0)),
        slippage_bps=float(costs.get("slippage_bps", 2.0)),
        fill_delay=int(cfg.get("fill_delay", 1)),
    )
    features = build_features(panel, lookback=lookback)
    weights = strategy.generate_weights(features)
    result = simulate(panel, weights, execution)
    report = compute(result, after_costs=True)

    if panel.origin is DataOrigin.SYNTHETIC:
        logs.append(SYNTHETIC_BANNER)
        logs.append(
            f"process={panel.metadata.process} seed={panel.metadata.seed} "
            f"n_symbols={panel.metadata.n_symbols} n_bars={panel.metadata.n_bars}"
        )
        logs.append(panel.metadata.notes)
    logs.append(
        "Execution: signal at t uses close[t]; fill at t+1 close after costs. "
        "This is a pipeline demo, not a trading signal."
    )
    logs.append(
        format_report(
            report,
            f"PIPELINE DEMO — {strategy.name} (full sample, not a hold-out result)",
        )
    )

    wf = cfg.get("walk_forward")
    if wf:
        dates = panel.dates
        splits = make_splits(
            dates,
            mode=str(wf.get("mode", "expanding")),
            train_bars=int(wf.get("train_bars", 252)),
            test_bars=int(wf.get("test_bars", 63)),
            step_bars=int(wf.get("step_bars", 63)),
            embargo_bars=int(wf.get("embargo_bars", 0)),
        )
        logs.append(
            f"Walk-forward folds: {len(splits)} "
            f"(mode={wf.get('mode', 'expanding')}, embargo={wf.get('embargo_bars', 0)}). "
            "Fold Sharpes are still not an edge — see RESEARCH.md."
        )
        for split in splits:
            test_panel = panel.copy_with_frame(
                panel.frame.filter(
                    (pl.col("date") >= split.test_start) & (pl.col("date") <= split.test_end)
                )
            )
            window = panel.copy_with_frame(
                panel.frame.filter(pl.col("date") <= split.test_end)
            )
            feat = build_features(window, lookback=lookback)
            w = strategy.generate_weights(feat)
            w = w.filter(
                (pl.col("date") >= split.test_start) & (pl.col("date") <= split.test_end)
            )
            if test_panel.frame.height == 0 or w.height == 0:
                continue
            sim = simulate(test_panel, w, execution)
            fold_report = compute(sim, after_costs=True)
            logs.append(
                format_report(
                    fold_report,
                    f"WF fold {split.fold} test {split.test_start} .. {split.test_end} "
                    f"(train ends {split.train_end})",
                )
            )
    if panel.origin is DataOrigin.SYNTHETIC:
        logs.append(SYNTHETIC_FOOTER)
    return report, logs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qre-backtest",
        description="Run a research backtest from a YAML config. Synthetic data is labeled.",
    )
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _, logs = run_backtest(cfg)
    sys.stdout.write("\n".join(logs) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
