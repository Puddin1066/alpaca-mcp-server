#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from alpaca_mcp_server.ml.score import score_dataset_rows


def main() -> None:
    p = argparse.ArgumentParser("Score options dataset rows")
    p.add_argument("--model", required=True, type=Path, help="Trained model .joblib")
    p.add_argument("--rows", required=True, type=Path, help="Rows file .csv.gz (same shape as training dataset)")
    p.add_argument("--out", required=True, type=Path, help="Output scored CSV")
    args = p.parse_args()

    out = score_dataset_rows(model_path=args.model, rows_csv_gz=args.rows, out_csv=args.out)
    print(str(out))


if __name__ == "__main__":
    main()

