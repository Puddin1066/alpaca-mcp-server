from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScoredOption:
    option_symbol: str
    score: float
    row: dict[str, Any]


def score_dataset_rows(
    *,
    model_path: Path,
    rows_csv_gz: Path,
    out_csv: Path,
) -> Path:
    """
    Score dataset-like rows with a trained model and write a CSV of scores.

    This expects the same feature columns as training.
    """
    try:
        import joblib  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing ML dependencies. Install with: python3 -m pip install -r requirements-ml.txt"
        ) from e

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(rows_csv_gz, compression="gzip")
    if df.empty:
        raise ValueError("Input rows are empty")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    proba = model.predict_proba(df[feature_cols])[:, 1]
    df_out = df.copy()
    df_out["score"] = proba
    df_out = df_out.sort_values("score", ascending=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    return out_csv

