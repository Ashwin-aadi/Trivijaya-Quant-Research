"""Build and print the known-artifacts register.

This is the review surface for the whole corporate-action layer: every point in the price series
that is unadjusted, unmodellable, or deliberately preserved, with the reason for each.

Usage:
    python scripts/show_artifacts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config  # noqa: E402
from src.common.io import write_derived_parquet  # noqa: E402
from src.data.artifacts import REGISTER_FILENAME, build_register  # noqa: E402


def main() -> int:
    cfg = load_config()
    register = build_register(cfg.paths.data_processed)
    write_derived_parquet(register, cfg.paths.data_processed / REGISTER_FILENAME)

    print(f"KNOWN-ARTIFACTS REGISTER — {register.height} entries "
          f"across {register['symbol'].n_unique()} symbols")
    print(f"stored at: {cfg.paths.data_processed / REGISTER_FILENAME}\n")
    for reason in register["reason"].unique().sort():
        subset = register.filter(register["reason"] == reason)
        print(f"[{reason}]  {subset.height} entr{'y' if subset.height == 1 else 'ies'}")
        for row in subset.iter_rows(named=True):
            window = (f"{row['start_date']}" if row["start_date"] == row["end_date"]
                      else f"{row['start_date']}..{row['end_date']}")
            print(f"   {row['symbol']:<12s} {window:<24s} {row['detail']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
