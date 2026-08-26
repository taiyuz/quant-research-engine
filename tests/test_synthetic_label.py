"""Synthetic generator and CLI must label data as SYNTHETIC. Not an edge."""

from __future__ import annotations

from qre.cli import format_report, run_backtest
from qre.data.loader import generate_synthetic_ohlcv
from qre.types import SYNTHETIC_BANNER, SYNTHETIC_FOOTER, DataOrigin


def test_generator_origin_is_synthetic() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=3, n_bars=40, seed=0, process="gbm")
    assert panel.metadata.origin is DataOrigin.SYNTHETIC
    assert panel.metadata.origin.value == "SYNTHETIC"
    assert "SYNTHETIC" in panel.metadata.notes
    assert panel.metadata.process == "gbm"


def test_ou_and_pair_are_labeled() -> None:
    ou, _ = generate_synthetic_ohlcv(n_symbols=2, n_bars=40, seed=1, process="ou")
    pair, _ = generate_synthetic_ohlcv(n_symbols=2, n_bars=40, seed=1, process="ou_pair")
    assert ou.origin is DataOrigin.SYNTHETIC
    assert pair.origin is DataOrigin.SYNTHETIC
    assert ou.metadata.process == "ou"
    assert pair.metadata.process == "ou_pair"


def test_cli_banner_and_footer() -> None:
    cfg = {
        "strategy": "momentum",
        "lookback": 10,
        "costs": {"commission_bps": 1.0, "slippage_bps": 2.0},
        "data": {"n_symbols": 4, "n_bars": 80, "seed": 42, "process": "gbm"},
    }
    report, logs = run_backtest(cfg)
    text = "\n".join(logs)
    assert SYNTHETIC_BANNER in text
    assert SYNTHETIC_FOOTER in text
    assert "NOT A MARKET RESULT" in text
    assert "Do not treat these metrics as evidence of an edge" in text
    assert report.data_origin is DataOrigin.SYNTHETIC
    assert "PIPELINE DEMO" in text
    rendered = format_report(report, "demo")
    assert "data_origin" in rendered
    assert "SYNTHETIC" in rendered


def test_banner_constants() -> None:
    assert SYNTHETIC_BANNER == "*** SYNTHETIC DATA — NOT A MARKET RESULT ***"
    assert "synthetic prices" in SYNTHETIC_FOOTER.lower()
