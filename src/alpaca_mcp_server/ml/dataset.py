from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .black_scholes import BlackScholesInputs, black_scholes_price
from .providers.base import AiFeatureProvider, Bar, MarketDataProvider, OptionContract, as_numeric_features


@dataclass(frozen=True)
class DatasetConfig:
    # Universe + time range
    underlying_symbols: Sequence[str]
    start: datetime
    end: datetime

    # Contract filters
    min_dte_days: int = 7
    max_dte_days: int = 45
    strikes_around_atm: int = 3  # use K around ATM rank (approx via spot)
    per_type_limit: int = 25

    # Labeling
    label_mode: str = "bs_underpriced"  # "bs_underpriced" | "future_return"
    risk_free_rate: float = 0.04
    realized_vol_lookback_days: int = 20
    underpriced_margin: float = 0.10  # market < (1 - margin)*fair => favorable
    future_horizon_days: int = 5
    future_return_threshold: float = 0.15


def _daily_close_series(bars: Sequence[Bar]) -> list[tuple[datetime, float]]:
    return [(b.timestamp, float(b.close)) for b in bars]


def _annualized_realized_vol_from_closes(closes: Sequence[float]) -> float | None:
    # close-to-close log returns annualized with 252 trading days
    if len(closes) < 3:
        return None
    rets: list[float] = []
    for a, b in zip(closes[:-1], closes[1:]):
        if a <= 0 or b <= 0:
            continue
        rets.append(math.log(b / a))
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252.0)


def _parse_option_symbol(option_symbol: str) -> tuple[str, date, str, float] | None:
    """
    Very common OCC-style symbol: ROOTYYMMDD[C|P]########
    Example: NVDA250919C00168000 (strike=168.000)
    """
    import re

    m = re.match(r"^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$", option_symbol)
    if not m:
        return None
    root, yy, mm, dd, cp, strike8 = m.groups()
    exp = date(2000 + int(yy), int(mm), int(dd))
    strike = int(strike8) / 1000.0
    opt_type = "call" if cp == "C" else "put"
    return root, exp, opt_type, strike


def build_options_dataset(
    *,
    provider: MarketDataProvider,
    cfg: DatasetConfig,
    out_csv_gz: Path,
    ai_provider: AiFeatureProvider | None = None,
) -> Path:
    """
    Build a supervised-learning dataset for "favorable pricing" detection.

    Output rows are per (underlying, option_contract, asof_date).

    Important notes / limitations:
    - If your data provider only supports *latest* option quotes (like the MCP server tools),
      then `label_mode="future_return"` is NOT a true historical backtest.
    - For true historical option modeling, your provider must be able to return option prices
      as-of the same timestamps you’re labeling.
    """
    import gzip

    out_csv_gz.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "asof",
        "underlying_symbol",
        "option_symbol",
        "expiration",
        "dte_days",
        "option_type",
        "strike",
        "spot",
        "moneyness",
        "rv_annualized",
        "option_mid",
        "label",
        "label_reason",
    ]
    # AI feature columns are dynamic; we’ll append them to each row.

    with gzip.open(out_csv_gz, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        # Snapshot dataset at `cfg.end` (as-of).
        #
        # We intentionally do NOT loop over historical as-of dates here because
        # this builder only requires a "latest option quote" API. Using a
        # latest-quote endpoint while iterating past timestamps would silently
        # produce incorrect (time-inconsistent) training data.
        asof_dt = cfg.end.astimezone(timezone.utc) if cfg.end.tzinfo else cfg.end.replace(tzinfo=timezone.utc)
        asof_date = asof_dt.date()

        # Iterate per underlying (single as-of).
        for sym in cfg.underlying_symbols:
            stock_bars = provider.get_stock_bars(sym, cfg.start, cfg.end, timeframe="1Day")
            closes = _daily_close_series(stock_bars)
            if len(closes) < cfg.realized_vol_lookback_days + 1:
                continue

            # Spot is latest close <= asof.
            _, spot = closes[-1]

            # realized vol from trailing window ending at asof
            window_closes = [c for _, c in closes[-cfg.realized_vol_lookback_days :]]
            rv = _annualized_realized_vol_from_closes(window_closes)
            if rv is None or rv <= 0:
                continue

            # contract universe: near-term expirations, strikes around ATM
            exp_gte = asof_date + timedelta(days=cfg.min_dte_days)
            exp_lte = asof_date + timedelta(days=cfg.max_dte_days)
            strike_gte = spot * 0.5
            strike_lte = spot * 1.5

            contracts = provider.get_option_contracts(
                underlying_symbol=sym,
                expiration_date_gte=exp_gte,
                expiration_date_lte=exp_lte,
                strike_price_gte=strike_gte,
                strike_price_lte=strike_lte,
                contract_type=None,
                limit=None,
            )
            if not contracts:
                continue

            # Keep strikes around ATM per type.
            calls = [c for c in contracts if c.contract_type == "call"]
            puts = [c for c in contracts if c.contract_type == "put"]

            def _keep_atm(contracts_: Sequence[OptionContract]) -> list[OptionContract]:
                ranked = sorted(contracts_, key=lambda c: abs(c.strike_price - spot))
                return ranked[: max(cfg.per_type_limit, cfg.strikes_around_atm * 2)]

            candidates = _keep_atm(calls) + _keep_atm(puts)

            ai_feats: dict[str, float] = {}
            if ai_provider is not None:
                ai_feats = dict(as_numeric_features(ai_provider.get_features(sym, asof_dt)))

            for c in candidates:
                dte = (c.expiration_date - asof_date).days
                if dte < cfg.min_dte_days or dte > cfg.max_dte_days:
                    continue

                option_mid = provider.get_option_latest_quote_mid(c.option_symbol)
                if option_mid is None or option_mid <= 0:
                    continue

                moneyness = spot / c.strike_price if c.strike_price > 0 else None
                if moneyness is None:
                    continue

                if cfg.label_mode == "future_return":
                    raise NotImplementedError(
                        "future_return labeling requires historical option prices (as-of). "
                        "Use a provider that can return option mid at historical timestamps."
                    )

                # bs_underpriced
                t_years = dte / 365.0
                fair = black_scholes_price(
                    BlackScholesInputs(
                        spot=spot,
                        strike=c.strike_price,
                        time_to_expiry_years=t_years,
                        rate=cfg.risk_free_rate,
                        volatility=rv,
                        option_type=c.contract_type,
                    )
                )
                threshold = (1.0 - cfg.underpriced_margin) * fair
                label = int(option_mid < threshold and fair > 0)
                label_reason = f"mid<{1.0 - cfg.underpriced_margin:.2f}*BS(rv); mid={option_mid:.4f} fair={fair:.4f}"

                row: dict[str, object] = {
                    "asof": asof_dt.isoformat(),
                    "underlying_symbol": sym,
                    "option_symbol": c.option_symbol,
                    "expiration": c.expiration_date.isoformat(),
                    "dte_days": dte,
                    "option_type": c.contract_type,
                    "strike": float(c.strike_price),
                    "spot": float(spot),
                    "moneyness": float(moneyness),
                    "rv_annualized": float(rv),
                    "option_mid": float(option_mid),
                    "label": int(label),
                    "label_reason": label_reason,
                }

                for k, v in ai_feats.items():
                    if k not in row:
                        row[k] = float(v)

                writer.writerow(row)

    return out_csv_gz

