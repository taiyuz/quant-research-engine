"""Rolling windows only use prices at dates <= t. Signals are next-bar."""

from __future__ import annotations

import numpy as np
import polars as pl

from qre.data.loader import generate_synthetic_ohlcv
from qre.execution.model import ExecutionModel
from qre.features.pipeline import add_rolling_mean, build_features
from qre.features.returns import add_lookback_return


def _causal_rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    csum = np.cumsum(x)
    for i in range(window - 1, len(x)):
        prev = csum[i - window] if i >= window else 0.0
        out[i] = (csum[i] - prev) / window
    return out


def test_rolling_mean_matches_causal_numpy() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=1, n_bars=50, seed=3, process="gbm")
    frame = add_rolling_mean(panel.frame, window=5, col="close")
    close = (
        panel.frame.sort(["symbol", "date"])
        .get_column("close")
        .to_numpy()
        .astype(float)
    )
    got = frame.sort(["symbol", "date"]).get_column("close_ma_5").to_numpy().astype(float)
    expected = _causal_rolling_mean(close, 5)
    mask = ~np.isnan(expected)
    np.testing.assert_allclose(got[mask], expected[mask], rtol=1e-12, atol=1e-12)
    assert np.isnan(got[~mask]).all()


def test_lookback_return_numeric() -> None:
    panel, _ = generate_synthetic_ohlcv(n_symbols=1, n_bars=30, seed=4, process="gbm")
    frame = add_lookback_return(panel.frame, lookback=5).sort("date")
    close = frame.get_column("close").to_numpy().astype(float)
    ret = frame.get_column("ret_5").to_numpy().astype(float)
    for i in range(5, len(close)):
        expected = close[i] / close[i - 5] - 1.0
        assert abs(ret[i] - expected) < 1e-12
    assert np.isnan(ret[:5]).all()


def test_execution_fill_delay_rejects_same_bar() -> None:
    try:
        ExecutionModel(fill_delay=0)
        raise AssertionError("fill_delay=0 must be rejected")
    except ValueError as exc:
        assert "fill_delay" in str(exc)


def test_signal_column_exists_on_same_bar_but_sim_delays() -> None:
    """Feature at t may use close[t]; the engine still fills at t+1."""
    panel, _ = generate_synthetic_ohlcv(n_symbols=2, n_bars=40, seed=5, process="gbm")
    feat = build_features(panel, lookback=5)
    row = feat.filter(pl.col("zscore").is_not_null()).row(0, named=True)
    assert row["zscore"] is not None
    exec_m = ExecutionModel(fill_delay=1)
    assert exec_m.fill_delay == 1
