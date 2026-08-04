"""Build the RegimeStress feature table for one frontier arm, using the frozen extractors.

P2 trained a model mapping *what a strategy is* --- concentration, holding period, turnover
profile, factor exposures --- onto how fragile it turns out to be, and reported that it does not
work: an out-of-sample R-squared of +0.024 against a mean baseline. This script is the first half
of asking whether that negative result survives a change of generator.

**Not one feature is redefined here.** ``concentration``, ``holding_period``,
``book_autocorrelation``, ``turnover_profile``, ``joint_betas`` and ``univariate_betas`` are
imported from :mod:`src.stress.characteristics`, and the factor regressors are P2's own standard
factors read from its frozen return panel, with the same knife-edge exclusion applied. A feature
computed differently for the two populations would make the comparison meaningless in exactly the
way the comparison exists to avoid.

Usage:
    python scripts/frontier_characteristics.py --arm gpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from build_characteristics import _factor_series  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.stress.characteristics import (  # noqa: E402
    book_autocorrelation,
    concentration,
    holding_period,
    joint_betas,
    turnover_profile,
    univariate_betas,
)

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
KNIFE_EDGE = ROOT / "benchmarks" / "regimestress" / "knife_edge.json"


def _books(positions: pl.DataFrame, name: str, sessions: list[object]) -> list[dict[str, float]]:
    """One dict per session in calendar order, cash sessions present as empty dicts.

    Built against the strategy's own session list rather than the dates present in the frame: a
    session holding nothing writes no rows, and inferring the list from the data would drop it and
    silently shorten every series computed from it.
    """
    own = positions.filter(pl.col("factor") == name)
    grouped: dict[object, dict[str, float]] = {}
    for row in own.iter_rows(named=True):
        grouped.setdefault(row["session_date"], {})[row["symbol"]] = row["weight"]
    return [grouped.get(day, {}) for day in sessions]


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    cfg = load_config()
    processed = cfg.paths.data_processed
    run = ROOT / "runs" / f"frontier_{args.arm}"

    # P2's own return panel supplies the factor regressors. The frontier strategies are the
    # subjects; the factors they are regressed against must stay exactly what P2 used.
    local_returns = pl.read_parquet(processed / "real_returns.parquet")
    knife = {
        r["name"] for r in json.loads(KNIFE_EDGE.read_text(encoding="utf-8"))["knife_edge"]
    }
    factors = _factor_series(local_returns, knife)
    factor_dates = local_returns.filter(
        pl.col("name") == next(iter(factors))
    ).sort("session_date")["session_date"].to_list()
    _log.info("%d factor regressors over %d sessions", len(factors), len(factor_dates))

    positions = pl.read_parquet(run / "positions.parquet")
    pooled = json.loads((run / "pooling.json").read_text(encoding="utf-8"))
    labels = [str(e["candidate"]).removesuffix(".py") for e in pooled["index"]]

    rows: list[dict[str, object]] = []
    for position, name in enumerate(labels):
        path = run / "backtests_development" / f"candidate_{position:03d}_returns.parquet"
        if not path.exists():
            _log.warning("no return series for %s; excluded from the feature table", name)
            continue
        own = pl.read_parquet(path).sort("session_date")
        sessions = own["session_date"].to_list()
        books = _books(positions, name, sessions)

        record: dict[str, object] = {
            "name": name,
            # Neither exclusion applies to a frontier arm: knife-edge and duplicate status were
            # frozen for the local corpus. They are carried as False so the column set matches the
            # table the model was trained on.
            "knife_edge": False,
            "duplicate": False,
            "n_sessions": len(sessions),
        }
        record.update(concentration(books))
        record.update(holding_period(books))
        record.update(book_autocorrelation(books))
        record.update(turnover_profile(own["turnover"].to_list()))

        shared = min(len(sessions), len(factor_dates))
        own_returns = own["return"].to_numpy()[:shared]
        aligned = {f: series[:shared] for f, series in factors.items()}
        record.update(joint_betas(own_returns, aligned))
        record.update({
            f"uni_{key}": value
            for key, value in univariate_betas(own_returns, aligned).items()
        })
        record["n_beta_sessions"] = shared
        rows.append(record)

    if not rows:
        raise SystemExit(f"no strategies with both a book and a return series for {args.arm}")

    table = pl.DataFrame(rows)
    out = run / "characteristics.parquet"
    table.write_parquet(out)
    _log.info("arm %s: %d strategies x %d columns -> %s",
              args.arm, table.height, table.width, out.relative_to(ROOT).as_posix())
    _log.info("  median holding period %.2f sessions, median turnover %.4f",
              float(np.nanmedian(table["mean_holding_period"].to_numpy().astype(float))),
              float(np.nanmedian(table["mean_turnover"].to_numpy().astype(float))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
