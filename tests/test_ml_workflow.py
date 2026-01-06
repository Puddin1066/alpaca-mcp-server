from __future__ import annotations

import gzip
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from alpaca_mcp_server.ml.dataset import DatasetConfig, build_options_dataset
from alpaca_mcp_server.ml.providers.mock import MockAiFeatureProvider, MockMarketDataProvider
from alpaca_mcp_server.ml.providers.base import Bar, OptionContract
from alpaca_mcp_server.ml.train import train_favorable_option_model
from alpaca_mcp_server.ml.score import score_dataset_rows


def _write_tmp(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_ml_workflow_end_to_end_with_mocks(tmp_path: Path) -> None:
    """
    End-to-end test of dataset -> train -> score using MOCK providers.

    IMPORTANT: This test does NOT call Alpaca APIs. It is fully offline.
    """
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 2, 1, tzinfo=timezone.utc)

    # Synthetic daily bars (enough to compute realized vol)
    bars: list[Bar] = []
    price = 100.0
    for i in range(40):
        ts = start + timedelta(days=i)
        # Vary returns so realized vol > 0 (avoid constant-return variance=0 edge case).
        price *= 1.001 if i % 2 == 0 else 1.003
        bars.append(
            Bar(
                timestamp=ts,
                open=price * 0.999,
                high=price * 1.002,
                low=price * 0.998,
                close=price,
                volume=1_000_000,
            )
        )

    asof_date = end.date()
    exp = asof_date + timedelta(days=30)
    contracts = [
        OptionContract(
            option_symbol="TEST250303C00100000",
            underlying_symbol="TEST",
            expiration_date=exp,
            strike_price=100.0,
            contract_type="call",
        ),
        OptionContract(
            option_symbol="TEST250303P00100000",
            underlying_symbol="TEST",
            expiration_date=exp,
            strike_price=100.0,
            contract_type="put",
        ),
    ]

    provider = MockMarketDataProvider(
        stock_bars_by_symbol={"TEST": bars},
        option_contracts_by_underlying={"TEST": contracts},
        option_mid_by_symbol={
            # Midprices chosen so at least one becomes labeled 1 (underpriced) under BS(rv) threshold.
            "TEST250303C00100000": 0.25,
            "TEST250303P00100000": 0.25,
        },
    )
    ai = MockAiFeatureProvider(features={"ai_sentiment": 0.7, "ai_uncertainty": 0.2})

    dataset_path = tmp_path / "dataset.csv.gz"
    cfg = DatasetConfig(underlying_symbols=["TEST"], start=start, end=end, underpriced_margin=0.10)
    build_options_dataset(provider=provider, cfg=cfg, out_csv_gz=dataset_path, ai_provider=ai)

    assert dataset_path.exists()

    df = pd.read_csv(dataset_path, compression="gzip")
    assert not df.empty
    assert {"ai_sentiment", "ai_uncertainty"}.issubset(set(df.columns))
    assert set(df["underlying_symbol"].unique()) == {"TEST"}
    assert set(df["option_symbol"].unique()) == {"TEST250303C00100000", "TEST250303P00100000"}
    assert df["label"].isin([0, 1]).all()

    model_path = tmp_path / "model.joblib"
    res = train_favorable_option_model(dataset_csv_gz=dataset_path, model_out=model_path)
    assert res.model_path.exists()
    assert res.n_rows == len(df)
    assert res.n_features > 0

    scored_csv = tmp_path / "scored.csv"
    score_dataset_rows(model_path=model_path, rows_csv_gz=dataset_path, out_csv=scored_csv)
    assert scored_csv.exists()

    scored = pd.read_csv(scored_csv)
    assert "score" in scored.columns
    assert scored["score"].between(0.0, 1.0).all()

