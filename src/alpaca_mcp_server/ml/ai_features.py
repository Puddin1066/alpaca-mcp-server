from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .providers.base import AiFeatureProvider, as_numeric_features


@dataclass(frozen=True)
class CsvAiFeatureProvider(AiFeatureProvider):
    """
    Loads "AI input" features from a CSV file.

    CSV requirements:
    - Must contain columns: underlying_symbol, asof
    - `asof` must be ISO datetime (e.g., 2025-01-02T15:30:00Z or 2025-01-02T15:30:00-05:00)
    - All other columns are treated as candidate numeric model features
    """

    csv_path: Path

    def __post_init__(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(str(self.csv_path))

    def get_features(self, underlying_symbol: str, asof: datetime) -> Mapping[str, float]:
        # Lazy import to keep ML deps optional.
        import pandas as pd  # type: ignore

        df = pd.read_csv(self.csv_path)
        if "underlying_symbol" not in df.columns or "asof" not in df.columns:
            raise ValueError("CSV must include columns: underlying_symbol, asof")

        df["asof"] = pd.to_datetime(df["asof"], utc=True, errors="coerce")
        if df["asof"].isna().any():
            raise ValueError("Invalid `asof` values; expected ISO datetimes")

        asof_utc = asof.astimezone(timezone.utc) if asof.tzinfo else asof.replace(tzinfo=timezone.utc)
        # Use the latest AI row at-or-before `asof` (more practical than exact timestamp matches).
        rows = df[(df["underlying_symbol"] == underlying_symbol) & (df["asof"] <= asof_utc)]
        if rows.empty:
            return {}

        row = rows.sort_values("asof").iloc[-1].to_dict()
        row.pop("underlying_symbol", None)
        row.pop("asof", None)
        return as_numeric_features(row)

