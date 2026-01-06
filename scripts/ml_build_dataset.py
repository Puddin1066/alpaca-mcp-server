#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from alpaca_mcp_server.ml.dataset import DatasetConfig, build_options_dataset
from alpaca_mcp_server.ml.providers.alpaca import AlpacaMarketDataProvider
from alpaca_mcp_server.ml.ai_features import CsvAiFeatureProvider


def _parse_iso_dt(s: str) -> datetime:
    # Accept YYYY-MM-DD as date shorthand (UTC midnight)
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return datetime.fromisoformat(s + "T00:00:00+00:00")
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _read_universe(path: Path) -> list[str]:
    syms: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        syms.append(line.upper())
    return syms


def main() -> None:
    p = argparse.ArgumentParser("Build biotech options ML dataset (CSV.gz)")
    p.add_argument("--universe", required=True, type=Path, help="Text file of underlying tickers, one per line")
    p.add_argument("--start", required=True, help="ISO datetime or YYYY-MM-DD")
    p.add_argument("--end", required=True, help="ISO datetime or YYYY-MM-DD")
    p.add_argument("--out", required=True, type=Path, help="Output dataset path (e.g. data/dataset.csv.gz)")
    p.add_argument("--ai-features", type=Path, default=None, help="Optional CSV with AI features (underlying_symbol, asof, ...)")
    p.add_argument("--min-dte", type=int, default=7)
    p.add_argument("--max-dte", type=int, default=45)
    p.add_argument("--underpriced-margin", type=float, default=0.10)
    p.add_argument("--risk-free-rate", type=float, default=0.04)
    args = p.parse_args()

    universe = _read_universe(args.universe)
    provider = AlpacaMarketDataProvider()
    ai = CsvAiFeatureProvider(args.ai_features) if args.ai_features else None

    cfg = DatasetConfig(
        underlying_symbols=universe,
        start=_parse_iso_dt(args.start),
        end=_parse_iso_dt(args.end),
        min_dte_days=args.min_dte,
        max_dte_days=args.max_dte,
        underpriced_margin=args.underpriced_margin,
        risk_free_rate=args.risk_free_rate,
        label_mode="bs_underpriced",
    )

    out = build_options_dataset(provider=provider, cfg=cfg, out_csv_gz=args.out, ai_provider=ai)
    print(str(out))


if __name__ == "__main__":
    main()

