"""OHLCV loaders and labeled synthetic generators.

Every path that invents prices stamps DataOrigin.SYNTHETIC on the panel.
GBM uses drift 0 (a martingale in expectation). OU is for mean-reversion
and pairs demos. Neither process is a market. Metrics computed on these
panels are pipeline tests, not alpha.

Processes
---------
gbm:      log S_{t+1} = log S_t + sigma * z_t          (mu = 0)
ou:       x_{t+1} = x_t + kappa * (theta - x_t) + sigma * z_t; S = exp(x)
ou_pair:  shared random-walk factor plus an OU spread so two names are
          cointegrated-like in logs (synthetic pair, not a listed spread).

A caller may request delist_mid_sample=True. The last symbol is then
removed after a mid-sample date. See qre.data.universe for PIT rules.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from qre.data.universe import MembershipRecord, PointInTimeUniverse
from qre.types import BarPanel, DataOrigin, PanelMetadata, OHLCV_COLUMNS


def _business_days(n: int, start: date = date(2020, 1, 2)) -> list[date]:
    days: list[date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _ohlc_from_close(close: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a plausible OHLC envelope around close. Not a microstructure model."""
    n = close.shape[0]
    open_px = np.empty_like(close)
    open_px[0] = close[0]
    open_px[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.0005, size=n - 1))
    noise = np.abs(rng.normal(0.0, 0.002, size=n))
    high = np.maximum(open_px, close) * (1.0 + noise)
    low = np.minimum(open_px, close) * (1.0 - noise)
    low = np.maximum(low, 1e-8)
    return open_px, high, low


def _frame_from_closes(
    dates: list[date],
    symbols: list[str],
    closes: np.ndarray,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """closes is (n_bars, n_symbols)."""
    n_bars, n_symbols = closes.shape
    rows: list[dict[str, object]] = []
    for j, symbol in enumerate(symbols):
        c = closes[:, j]
        o, h, l = _ohlc_from_close(c, rng)
        vol = rng.integers(100_000, 1_000_000, size=n_bars)
        for i in range(n_bars):
            rows.append(
                {
                    "date": dates[i],
                    "symbol": symbol,
                    "open": float(o[i]),
                    "high": float(h[i]),
                    "low": float(l[i]),
                    "close": float(c[i]),
                    "volume": int(vol[i]),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).select(list(OHLCV_COLUMNS))


def generate_synthetic_ohlcv(
    n_symbols: int = 8,
    n_bars: int = 504,
    seed: int = 42,
    process: str = "gbm",
    delist_mid_sample: bool = False,
    sigma: float = 0.012,
    kappa: float = 0.08,
    theta: float = 0.0,
) -> tuple[BarPanel, PointInTimeUniverse]:
    """Return a SYNTHETIC panel and its point-in-time universe.

    Drift is 0 for GBM. Do not raise drift to manufacture a Sharpe.
    """
    if n_symbols < 1:
        raise ValueError("n_symbols must be >= 1")
    if n_bars < 2:
        raise ValueError("n_bars must be >= 2")
    process = process.lower()
    if process not in {"gbm", "ou", "ou_pair"}:
        raise ValueError(f"unknown process {process!r}; expected gbm, ou, or ou_pair")
    if process == "ou_pair" and n_symbols < 2:
        raise ValueError("ou_pair requires at least 2 symbols")

    rng = np.random.default_rng(seed)
    dates = _business_days(n_bars)
    symbols = [f"SYN{i:03d}" for i in range(n_symbols)]
    closes = np.zeros((n_bars, n_symbols), dtype=float)

    if process == "gbm":
        z = rng.normal(0.0, 1.0, size=(n_bars, n_symbols))
        log_r = sigma * z
        log_s = np.cumsum(log_r, axis=0) + np.log(100.0)
        closes = np.exp(log_s)
    elif process == "ou":
        x = np.zeros((n_bars, n_symbols), dtype=float)
        x[0] = rng.normal(theta, sigma / max(kappa, 1e-6), size=n_symbols)
        for t in range(1, n_bars):
            x[t] = x[t - 1] + kappa * (theta - x[t - 1]) + sigma * rng.normal(0.0, 1.0, size=n_symbols)
        closes = np.exp(x + np.log(100.0))
    else:
        factor = np.cumsum(sigma * rng.normal(0.0, 1.0, size=n_bars))
        spread = np.zeros(n_bars, dtype=float)
        for t in range(1, n_bars):
            spread[t] = spread[t - 1] + kappa * (0.0 - spread[t - 1]) + 0.5 * sigma * rng.normal()
        closes[:, 0] = np.exp(np.log(100.0) + factor)
        closes[:, 1] = np.exp(np.log(100.0) + factor + spread)
        if n_symbols > 2:
            z = rng.normal(0.0, 1.0, size=(n_bars, n_symbols - 2))
            closes[:, 2:] = np.exp(np.log(100.0) + np.cumsum(sigma * z, axis=0))

    frame = _frame_from_closes(dates, symbols, closes, rng)

    records: list[MembershipRecord] = []
    delist_date: date | None = None
    delist_symbol: str | None = None
    if delist_mid_sample and n_symbols >= 1:
        cut = max(n_bars // 2, 10)
        delist_date = dates[cut]
        delist_symbol = symbols[-1]
        frame = frame.filter(
            ~((pl.col("symbol") == delist_symbol) & (pl.col("date") > delist_date))
        )

    for symbol in symbols:
        start = dates[0]
        ddate = delist_date if symbol == delist_symbol else None
        records.append(MembershipRecord(symbol=symbol, start_date=start, delist_date=ddate))

    universe = PointInTimeUniverse(records=tuple(records))
    notes = (
        "SYNTHETIC prices. GBM drift is 0; OU/pairs are toy mean-reverting "
        "processes. Sample backtest metrics are a pipeline demo, not alpha."
    )
    if delist_symbol is not None:
        notes += f" {delist_symbol} delisted on {delist_date} (inclusive last bar)."

    panel = BarPanel(
        frame=frame.sort(["symbol", "date"]),
        metadata=PanelMetadata(
            origin=DataOrigin.SYNTHETIC,
            process=process,
            seed=seed,
            n_symbols=n_symbols,
            n_bars=n_bars,
            notes=notes,
            extra={"delist_symbol": delist_symbol or "", "delist_date": str(delist_date or "")},
        ),
    )
    universe.assert_no_post_delist_bars(panel)
    return panel, universe


def load_ohlcv_csv(path: str | Path, origin: DataOrigin = DataOrigin.USER_PROVIDED) -> BarPanel:
    frame = pl.read_csv(path, try_parse_dates=True)
    frame = frame.rename({c: c.lower() for c in frame.columns})
    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"csv missing columns {missing}")
    frame = frame.select(list(OHLCV_COLUMNS)).with_columns(pl.col("date").cast(pl.Date))
    return BarPanel(
        frame=frame.sort(["symbol", "date"]),
        metadata=PanelMetadata(origin=origin, notes=f"loaded csv {path}"),
    )


def load_ohlcv_parquet(path: str | Path, origin: DataOrigin = DataOrigin.USER_PROVIDED) -> BarPanel:
    frame = pl.read_parquet(path)
    frame = frame.rename({c: c.lower() for c in frame.columns})
    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"parquet missing columns {missing}")
    frame = frame.select(list(OHLCV_COLUMNS)).with_columns(pl.col("date").cast(pl.Date))
    return BarPanel(
        frame=frame.sort(["symbol", "date"]),
        metadata=PanelMetadata(origin=origin, notes=f"loaded parquet {path}"),
    )
