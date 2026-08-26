"""Shared panel types, origin labels, and research-convention constants.

OHLCV columns are always: date, symbol, open, high, low, close, volume.
Dates are chronological per symbol. DataOrigin.SYNTHETIC is not optional
decoration — every report, CLI print, and metric struct must carry it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Final

import polars as pl

OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

TRADING_DAYS_PER_YEAR: Final[int] = 252

SYNTHETIC_BANNER: Final[str] = "*** SYNTHETIC DATA — NOT A MARKET RESULT ***"
SYNTHETIC_FOOTER: Final[str] = (
    "This run used labeled synthetic prices. "
    "Do not treat these metrics as evidence of an edge."
)


class DataOrigin(str, Enum):
    """Where the price panel came from.

    SYNTHETIC: generated in-process (GBM / OU). Never an edge claim.
    USER_PROVIDED: loaded from csv/parquet the caller supplied.
    """

    SYNTHETIC = "SYNTHETIC"
    USER_PROVIDED = "USER_PROVIDED"


@dataclass(frozen=True)
class PanelMetadata:
    origin: DataOrigin
    process: str | None = None
    seed: int | None = None
    n_symbols: int | None = None
    n_bars: int | None = None
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class BarPanel:
    """Point-in-time OHLCV panel plus origin metadata.

    `frame` must contain OHLCV_COLUMNS. Rows for a delisted name must not
    appear after that name's delist_date (see qre.data.universe).
    """

    frame: pl.DataFrame
    metadata: PanelMetadata

    def __post_init__(self) -> None:
        missing = [c for c in OHLCV_COLUMNS if c not in self.frame.columns]
        if missing:
            raise ValueError(f"BarPanel missing required columns: {missing}")

    @property
    def origin(self) -> DataOrigin:
        return self.metadata.origin

    @property
    def dates(self) -> list[date]:
        return sorted(self.frame.get_column("date").unique().to_list())

    @property
    def symbols(self) -> list[str]:
        return sorted(self.frame.get_column("symbol").unique().to_list())

    def copy_with_frame(self, frame: pl.DataFrame) -> BarPanel:
        return BarPanel(frame=frame, metadata=self.metadata)
