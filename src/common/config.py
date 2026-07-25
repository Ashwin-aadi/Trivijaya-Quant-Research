"""Load config/config.yaml into a typed, hashable configuration object.

Two jobs:
  1. Give the rest of the code typed, dot-accessed parameters instead of a loose dict, so a typo
     in a key fails at load time rather than deep inside a backtest.
  2. Produce a stable ``config_hash`` that goes into every run manifest — two runs with the same
     hash used the same parameters, which is what reproducibility actually means here.

Sections are modelled as they are introduced. Values not yet modelled stay reachable through
``Config.raw`` so an unrecognised key never silently blocks a run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.common.exceptions import ConfigError

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


@dataclass(frozen=True)
class MetaConfig:
    seed: int
    project: str


@dataclass(frozen=True)
class PathsConfig:
    data_raw: Path
    data_interim: Path
    data_processed: Path
    runs: Path


@dataclass(frozen=True)
class CalendarConfig:
    exchange: str
    history_start: date


@dataclass(frozen=True)
class DatesConfig:
    dev_start: date
    dev_end: date
    holdout_start: date


@dataclass(frozen=True)
class UniverseConfig:
    index: str
    membership_source: str | None
    min_median_adv_inr: int


@dataclass(frozen=True)
class PricesConfig:
    authoritative_source: str | None
    cross_check_rel_tol: float


@dataclass(frozen=True)
class DataConfig:
    prices: PricesConfig


@dataclass(frozen=True)
class Config:
    meta: MetaConfig
    paths: PathsConfig
    calendar: CalendarConfig
    dates: DatesConfig
    universe: UniverseConfig
    data: DataConfig
    raw: dict[str, Any]        # everything as loaded, including keys not yet modelled
    source_path: Path
    config_hash: str           # sha256 of the raw file bytes


def _require(section: dict[str, Any], key: str, where: str) -> Any:  # noqa: ANN401
    # Returns a raw YAML value whose type isn't known until the caller validates it.
    """Fetch a required key, raising ConfigError with a locating message if it is absent."""
    if key not in section:
        raise ConfigError(f"missing required key '{key}' in config section '{where}'")
    return section[key]


def _parse_date(value: Any, where: str) -> date:  # noqa: ANN401
    # `value` is a raw YAML scalar (str, date, or something invalid we reject below).
    """Accept an ISO date string or an already-parsed date; reject anything else."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"bad date '{value}' at {where}: {exc}") from exc
    raise ConfigError(f"expected an ISO date at {where}, got {type(value).__name__}")


def _build(raw: dict[str, Any], source_path: Path, config_hash: str) -> Config:
    """Assemble a Config from the raw mapping, validating each section as it is read."""
    meta = _require(raw, "meta", "root")
    paths = _require(raw, "paths", "root")
    cal = _require(raw, "calendar", "root")
    dates = _require(raw, "dates", "root")
    uni = _require(raw, "universe", "root")
    prices = _require(_require(raw, "data", "root"), "prices", "data")

    return Config(
        meta=MetaConfig(seed=int(_require(meta, "seed", "meta")),
                        project=str(_require(meta, "project", "meta"))),
        paths=PathsConfig(
            data_raw=Path(_require(paths, "data_raw", "paths")),
            data_interim=Path(_require(paths, "data_interim", "paths")),
            data_processed=Path(_require(paths, "data_processed", "paths")),
            runs=Path(_require(paths, "runs", "paths")),
        ),
        calendar=CalendarConfig(
            exchange=str(_require(cal, "exchange", "calendar")),
            history_start=_parse_date(_require(cal, "history_start", "calendar"),
                                      "calendar.history_start"),
        ),
        dates=DatesConfig(
            dev_start=_parse_date(_require(dates, "dev_start", "dates"), "dates.dev_start"),
            dev_end=_parse_date(_require(dates, "dev_end", "dates"), "dates.dev_end"),
            holdout_start=_parse_date(_require(dates, "holdout_start", "dates"),
                                      "dates.holdout_start"),
        ),
        universe=UniverseConfig(
            index=str(_require(uni, "index", "universe")),
            membership_source=uni.get("membership_source"),
            min_median_adv_inr=int(_require(uni, "min_median_adv_inr", "universe")),
        ),
        data=DataConfig(prices=PricesConfig(
            authoritative_source=prices.get("authoritative_source"),
            cross_check_rel_tol=float(_require(prices, "cross_check_rel_tol", "data.prices")),
        )),
        raw=raw,
        source_path=source_path,
        config_hash=config_hash,
    )


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Read and validate the YAML config, returning a typed Config with a content hash."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    config_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    # Sanity: the holdout must start after the dev window ends, or the split is meaningless.
    cfg = _build(raw, path, config_hash)
    if cfg.dates.holdout_start <= cfg.dates.dev_end:
        raise ConfigError(
            f"holdout_start ({cfg.dates.holdout_start}) must be after dev_end ({cfg.dates.dev_end})"
        )
    return cfg
