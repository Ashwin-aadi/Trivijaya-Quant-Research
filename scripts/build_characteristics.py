"""Assemble the Phase 2.2 feature table: what each strategy *is*, independent of how it is stressed.

Reads the books recorded by ``scripts/persist_positions.py``, the turnover series recorded beside
them, and the realised returns from ``scripts/persist_real_returns.py``. Produces one row per
strategy in ``data/processed/characteristics.parquet``.

Nothing here touches a synthetic path or the fragility target, so no feature can leak the quantity
the predictor is asked to estimate.

Factor exposures are **joint** betas — one multiple regression per strategy against all usable
factor proxies at once, per the PI ruling of 2026-08-02. Univariate betas are computed alongside
and stored with a ``uni_`` prefix as a reported sensitivity, so the effect of the specification
choice is a measured quantity rather than an argument. The design's condition number is written
to the run manifest and to ``data/processed/factor_design.json``.

Two exclusions apply, both for reasons that would otherwise corrupt the feature rather than merely
weaken it:

* a knife-edge factor is not used as a regressor, because its own return series is not a stable
  function of its inputs, so a beta against it would inherit that instability;
* a factor's beta against **itself** is recorded as null rather than 1.0, since a feature that is
  exactly one for a single row is an identifier, not a characteristic.

Usage:
    python scripts/build_characteristics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from freeze_knife_edge import FACTORS  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.stress.characteristics import (  # noqa: E402
    book_autocorrelation,
    concentration,
    design_diagnostics,
    holding_period,
    joint_betas,
    turnover_profile,
    univariate_betas,
)

_log = get_logger(__name__)

POSITIONS = Path("data/interim/positions")
KNIFE_EDGE = Path("benchmarks/regimestress/knife_edge.json")
DUPLICATES = Path("benchmarks/regimestress/duplicates.json")


def _books(name: str, sessions: list[object]) -> list[dict[str, float]]:
    """One dict per session, in calendar order, with cash sessions present as empty dicts.

    Built against the strategy's own session list rather than against whatever dates appear in the
    parquet: a session on which nothing was held writes no rows, and inferring the session list from
    the file would silently drop it and shorten the series.
    """
    frame = pl.read_parquet(POSITIONS / f"{name}.parquet")
    grouped: dict[object, dict[str, float]] = {}
    for row in frame.iter_rows(named=True):
        grouped.setdefault(row["session_date"], {})[row["symbol"]] = row["weight"]
    return [grouped.get(day, {}) for day in sessions]


def _factor_series(returns: pl.DataFrame, knife: set[str]) -> dict[str, np.ndarray]:
    """Net return series of each usable standard factor, aligned on session_date."""
    usable = sorted(FACTORS - knife)
    dropped = sorted(FACTORS & knife)
    if dropped:
        _log.warning("excluded %d knife-edge factor(s) as regressors: %s", len(dropped), dropped)
    wide = returns.filter(pl.col("name").is_in(usable)).pivot(
        on="name", index="session_date", values="net_return"
    ).sort("session_date")
    return {name: wide[name].to_numpy() for name in usable if name in wide.columns}


def main() -> int:
    configure_logging()
    cfg = load_config()
    processed = cfg.paths.data_processed

    returns = pl.read_parquet(processed / "real_returns.parquet")
    turnover = pl.read_parquet(processed / "session_turnover.parquet")
    knife = {
        r["name"] for r in
        json.loads(KNIFE_EDGE.read_text(encoding="utf-8"))["knife_edge"]
    }
    duplicate = set(json.loads(DUPLICATES.read_text(encoding="utf-8"))["removed"])
    factors = _factor_series(returns, knife)
    factor_dates = returns.filter(
        pl.col("name") == next(iter(factors))
    ).sort("session_date")["session_date"].to_list()
    _log.info("%d factor regressors over %d sessions", len(factors), len(factor_dates))

    with RunManifest(cfg, script="build_characteristics.py") as run:
        rows: list[dict[str, object]] = []
        for name in returns["name"].unique(maintain_order=True).to_list():
            if not (POSITIONS / f"{name}.parquet").exists():
                _log.warning("no recorded book for %s; excluded from the feature table", name)
                continue
            own = returns.filter(pl.col("name") == name).sort("session_date")
            sessions = own["session_date"].to_list()
            books = _books(name, sessions)

            record: dict[str, object] = {"name": name, "knife_edge": name in knife,
                                         "duplicate": name in duplicate,
                                         "n_sessions": len(sessions)}
            record.update(concentration(books))
            record.update(holding_period(books))
            record.update(book_autocorrelation(books))
            record.update(turnover_profile(
                turnover.filter(pl.col("name") == name).sort("session_index")["turnover"].to_list()
            ))

            # Betas need the strategy and each factor on the same sessions. Strategies that were
            # ruined early have shorter series, so both sides are cut to the shared prefix rather
            # than assumed aligned.
            shared = min(len(sessions), len(factor_dates))
            own_returns = own["net_return"].to_numpy()[:shared]
            aligned = {f: series[:shared] for f, series in factors.items()}
            betas = joint_betas(own_returns, aligned)
            univariate = {
                f"uni_{key}": value
                for key, value in univariate_betas(own_returns, aligned).items()
            }
            if name in factors:
                # A strategy regressed on itself returns a coefficient of exactly 1.0 and zeros
                # elsewhere. That is an identifier for one row, not a characteristic.
                betas = {key: float("nan") for key in betas}
                univariate[f"uni_beta_{name}"] = float("nan")
            record.update(betas)
            record.update(univariate)
            record["n_beta_sessions"] = shared
            rows.append(record)

        design = design_diagnostics(factors)
        (processed / "factor_design.json").write_text(
            json.dumps(design, indent=2, sort_keys=True), encoding="utf-8"
        )
        _log.info(
            "factor design: %d regressors, condition number %.1f, max pairwise |corr| %.3f",
            int(design["n_factors"]), design["condition_number"],
            design["max_abs_pairwise_correlation"],
        )
        run.note("factor_condition_number", design["condition_number"])

        table = pl.DataFrame(rows).sort("name")
        out = processed / "characteristics.parquet"
        table.write_parquet(out)
        run.note("strategies", table.height)
        run.note("features", len(table.columns))

    feature_columns = [c for c in table.columns if c not in {"name", "knife_edge"}]
    _log.info("%d strategies x %d features -> %s", table.height, len(feature_columns), out)
    _log.info("  knife-edge flagged: %d   duplicate flagged: %d",
              int(table["knife_edge"].sum()), int(table["duplicate"].sum()))
    for column in ("mean_holding_period", "effective_holdings", "mean_turnover",
                   "book_similarity_21d"):
        values = table[column].drop_nulls().drop_nans().to_numpy()
        _log.info(
            "  %-22s median %8.3f   range %8.3f .. %8.3f   (n=%d)",
            column, float(np.median(values)), float(values.min()), float(values.max()),
            values.size,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
