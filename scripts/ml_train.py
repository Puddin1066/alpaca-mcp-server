#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from alpaca_mcp_server.ml.train import train_favorable_option_model


def main() -> None:
    p = argparse.ArgumentParser("Train favorable option model")
    p.add_argument("--dataset", required=True, type=Path, help="Input dataset .csv.gz")
    p.add_argument("--out", required=True, type=Path, help="Output model path (e.g. models/model.joblib)")
    args = p.parse_args()

    res = train_favorable_option_model(dataset_csv_gz=args.dataset, model_out=args.out)
    print(f"model={res.model_path}")
    print(f"rows={res.n_rows} features={res.n_features}")
    print(f"metrics={res.metrics}")


if __name__ == "__main__":
    main()

