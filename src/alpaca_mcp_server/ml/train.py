from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics: dict[str, float]
    n_rows: int
    n_features: int


def train_favorable_option_model(
    *,
    dataset_csv_gz: Path,
    model_out: Path,
    time_column: str = "asof",
    target_column: str = "label",
) -> TrainResult:
    """
    Train a simple supervised model to predict `label` from dataset features.

    Requires optional ML deps: pandas, scikit-learn, joblib.
    """
    try:
        import joblib  # type: ignore
        import pandas as pd  # type: ignore
        from sklearn.compose import ColumnTransformer  # type: ignore
        from sklearn.impute import SimpleImputer  # type: ignore
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing ML dependencies. Install with: python3 -m pip install -r requirements-ml.txt"
        ) from e

    df = pd.read_csv(dataset_csv_gz, compression="gzip")
    if df.empty:
        raise ValueError("Dataset is empty")
    if time_column not in df.columns or target_column not in df.columns:
        raise ValueError(f"Dataset must include columns: {time_column}, {target_column}")

    df[time_column] = pd.to_datetime(df[time_column], utc=True, errors="coerce")
    df = df.dropna(subset=[time_column, target_column])
    df = df.sort_values(time_column)

    y = df[target_column].astype(int)

    # Feature set: numeric columns excluding identifiers.
    drop_cols = {
        target_column,
        time_column,
        "underlying_symbol",
        "option_symbol",
        "expiration",
        "label_reason",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    numeric_cols = list(X.columns)

    # Time-based split (last 20% of rows is test).
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ]
                ),
                numeric_cols,
            )
        ],
        remainder="drop",
    )

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=250,
        random_state=42,
    )

    model = Pipeline(steps=[("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)

    # Metrics
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else float("nan"),
        "avg_precision": float(average_precision_score(y_test, proba)) if len(set(y_test)) > 1 else float("nan"),
        "test_rows": float(len(y_test)),
        "train_rows": float(len(y_train)),
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "time_column": time_column,
            "target_column": target_column,
            "metrics": metrics,
        },
        model_out,
    )

    return TrainResult(
        model_path=model_out,
        metrics=metrics,
        n_rows=int(len(df)),
        n_features=int(len(feature_cols)),
    )

