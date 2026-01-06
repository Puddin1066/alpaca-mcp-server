from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class OptionContract:
    option_symbol: str
    underlying_symbol: str
    expiration_date: date
    strike_price: float
    contract_type: str  # "call" or "put"


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider(Protocol):
    """Read-only market data provider interface (strategy research)."""

    def get_stock_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> Sequence[Bar]:
        ...

    def get_option_contracts(
        self,
        underlying_symbol: str,
        expiration_date_gte: date | None = None,
        expiration_date_lte: date | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
        contract_type: str | None = None,  # "call"|"put"|None
        limit: int | None = None,
    ) -> Sequence[OptionContract]:
        ...

    def get_option_latest_quote_mid(self, option_symbol: str) -> float | None:
        """Return midprice (bid+ask)/2 if available."""
        ...


class AiFeatureProvider(Protocol):
    """
    External "AI input" feature provider.

    This should return *numeric* features (already engineered) for a given underlying
    and timestamp. Example features: sentiment score, catalyst probability, uncertainty.
    """

    def get_features(
        self, underlying_symbol: str, asof: datetime
    ) -> Mapping[str, float]:
        ...


def as_numeric_features(d: Mapping[str, Any]) -> dict[str, float]:
    """Filter/convert mapping to float-only feature dict (drops non-numeric)."""
    out: dict[str, float] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
            continue
        # allow numeric strings
        if isinstance(v, str):
            try:
                out[str(k)] = float(v)
            except ValueError:
                continue
    return out

