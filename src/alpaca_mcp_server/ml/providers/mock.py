from __future__ import annotations

"""
MOCK provider implementations for offline tests.

Important: these providers DO NOT call Alpaca and will return synthetic data.
Keep this explicit so it's always clear when API calls are being mocked.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Mapping, Sequence

from .base import AiFeatureProvider, Bar, MarketDataProvider, OptionContract


@dataclass(frozen=True)
class MockAiFeatureProvider(AiFeatureProvider):
    features: Mapping[str, float]

    def get_features(self, underlying_symbol: str, asof: datetime) -> Mapping[str, float]:
        _ = (underlying_symbol, asof)
        return dict(self.features)


@dataclass
class MockMarketDataProvider(MarketDataProvider):
    stock_bars_by_symbol: dict[str, Sequence[Bar]]
    option_contracts_by_underlying: dict[str, Sequence[OptionContract]]
    option_mid_by_symbol: dict[str, float]

    def get_stock_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str
    ) -> Sequence[Bar]:
        _ = (start, end, timeframe)
        return list(self.stock_bars_by_symbol.get(symbol, []))

    def get_option_contracts(
        self,
        underlying_symbol: str,
        expiration_date_gte: date | None = None,
        expiration_date_lte: date | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
        contract_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[OptionContract]:
        contracts = list(self.option_contracts_by_underlying.get(underlying_symbol, []))
        if contract_type:
            contracts = [c for c in contracts if c.contract_type == contract_type]
        if expiration_date_gte:
            contracts = [c for c in contracts if c.expiration_date >= expiration_date_gte]
        if expiration_date_lte:
            contracts = [c for c in contracts if c.expiration_date <= expiration_date_lte]
        if strike_price_gte is not None:
            contracts = [c for c in contracts if c.strike_price >= strike_price_gte]
        if strike_price_lte is not None:
            contracts = [c for c in contracts if c.strike_price <= strike_price_lte]
        if limit is not None:
            contracts = contracts[:limit]
        return contracts

    def get_option_latest_quote_mid(self, option_symbol: str) -> float | None:
        return self.option_mid_by_symbol.get(option_symbol)

