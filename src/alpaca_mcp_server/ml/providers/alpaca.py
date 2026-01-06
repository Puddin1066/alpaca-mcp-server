from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Sequence

from dotenv import load_dotenv

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    OptionLatestQuoteRequest,
    OptionChainRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import OptionsFeed
from alpaca.trading.enums import ContractType

from ..dataset import _parse_option_symbol
from .base import Bar, MarketDataProvider, OptionContract


def _timeframe_from_str(timeframe: str) -> TimeFrame:
    # Minimal set used by our dataset builder.
    tf = timeframe.strip()
    if tf == "1Day":
        return TimeFrame.Day
    if tf == "1Hour":
        return TimeFrame.Hour
    if tf == "1Min":
        return TimeFrame.Minute
    raise ValueError("Unsupported timeframe. Use one of: 1Min, 1Hour, 1Day")


@dataclass
class AlpacaMarketDataProvider(MarketDataProvider):
    """
    Alpaca-backed provider for strategy research.

    Uses env vars:
    - ALPACA_API_KEY
    - ALPACA_SECRET_KEY
    """

    feed: OptionsFeed | None = None
    _stock: StockHistoricalDataClient = field(init=False)
    _opt: OptionHistoricalDataClient = field(init=False)
    _quote_mid_cache: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        load_dotenv()
        import os

        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in environment")

        self._stock = StockHistoricalDataClient(key, secret)
        self._opt = OptionHistoricalDataClient(api_key=key, secret_key=secret)

    def get_stock_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str
    ) -> Sequence[Bar]:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_timeframe_from_str(timeframe),
            start=start,
            end=end,
            limit=10000,
        )
        resp = self._stock.get_stock_bars(req)
        bars = resp.data.get(symbol, [])
        out: list[Bar] = []
        for b in bars:
            ts = b.timestamp
            ts_utc = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            out.append(
                Bar(
                    timestamp=ts_utc,
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume),
                )
            )
        return out

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
        type_enum = None
        if contract_type == "call":
            type_enum = ContractType.CALL
        elif contract_type == "put":
            type_enum = ContractType.PUT
        elif contract_type is None:
            type_enum = None
        else:
            raise ValueError("contract_type must be 'call', 'put', or None")

        req = OptionChainRequest(
            underlying_symbol=underlying_symbol,
            feed=self.feed,
            type=type_enum,
            strike_price_gte=strike_price_gte,
            strike_price_lte=strike_price_lte,
            expiration_date_gte=expiration_date_gte,
            expiration_date_lte=expiration_date_lte,
            limit=limit,
        )

        chain = self._opt.get_option_chain(req) or {}
        out: list[OptionContract] = []
        for option_symbol in chain.keys():
            parsed = _parse_option_symbol(option_symbol)
            if not parsed:
                # Skip unknown formats rather than failing dataset builds.
                continue
            root, exp, opt_type, strike = parsed
            out.append(
                OptionContract(
                    option_symbol=option_symbol,
                    underlying_symbol=underlying_symbol,
                    expiration_date=exp,
                    strike_price=float(strike),
                    contract_type=opt_type,
                )
            )

        return out

    def get_option_latest_quote_mid(self, option_symbol: str) -> float | None:
        if option_symbol in self._quote_mid_cache:
            return self._quote_mid_cache[option_symbol]

        req = OptionLatestQuoteRequest(symbol_or_symbols=option_symbol, feed=self.feed)
        quotes = self._opt.get_option_latest_quote(req)
        q = quotes.get(option_symbol) if quotes else None
        if not q:
            return None

        bid = float(q.bid_price or 0.0)
        ask = float(q.ask_price or 0.0)
        if bid <= 0 or ask <= 0:
            return None

        mid = 0.5 * (bid + ask)
        self._quote_mid_cache[option_symbol] = mid
        return mid

