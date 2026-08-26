"""Quantitative trading research engine: strategies, execution costs, and anti-overfitting guardrails."""
from __future__ import annotations

from qre.types import BarPanel, DataOrigin, PanelMetadata, SYNTHETIC_BANNER, SYNTHETIC_FOOTER

__version__ = "0.1.0"
__all__ = [
    "BarPanel",
    "DataOrigin",
    "PanelMetadata",
    "SYNTHETIC_BANNER",
    "SYNTHETIC_FOOTER",
    "__version__",
]

