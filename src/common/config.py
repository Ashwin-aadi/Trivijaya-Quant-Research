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
    index_symbol: str            # index whose level history defines the trading sessions


@dataclass(frozen=True)
class DatesConfig:
    dev_start: date
    dev_end: date
    holdout_start: date
    #: Frozen before any holdout data was fetched. A holdout whose end moves with whatever data
    #: happens to be available is not a fixed window, and its length becomes a free parameter.
    holdout_end: date


@dataclass(frozen=True)
class UniverseConfig:
    """Parameters defining the rules-based liquidity universe.

    Every field after ``size`` is a methodological choice awaiting PI ratification, not an
    implementation detail — see config.yaml for the reasoning behind each proposed default.
    """

    method: str
    size: int
    trailing_sessions: int       # lookback over which median traded value is measured
    rebalance: str               # "quarterly" | "monthly"
    entry_rank: int              # must rank at least this high to join
    exit_rank: int               # incumbent is dropped only once it falls past this
    min_listed_sessions: int     # required listed history before a name is eligible
    min_traded_fraction: float   # required share of the window on which it actually traded


@dataclass(frozen=True)
class PricesConfig:
    authoritative_source: str | None
    cross_check_rel_tol: float
    max_discrepancy_rate: float  # above this, halt rather than proceed (broken adjustment)


@dataclass(frozen=True)
class DataConfig:
    prices: PricesConfig


@dataclass(frozen=True)
class AuditConfig:
    """Parameters for the semantic audit layer's local-model calls.

    ``model_tag`` is reproducibility-load-bearing rather than a tuning knob: a semantic label is
    only reproducible against a pinned tag and quantization, so it is recorded on every finding
    and in the run manifest. See config.yaml for why ``num_ctx`` is pinned too.
    """

    model_tag: str
    ollama_host: str
    num_ctx: int                    # pinned: a smaller window truncates the prompt silently
    request_timeout_seconds: float
    probe_timeout_seconds: float    # reachability check only, so it fails fast


@dataclass(frozen=True)
class SegmentRates:
    """Rates that differ between delivery and intraday, and between buy and sell."""

    stt_buy: float
    stt_sell: float
    stamp_duty_buy: float
    stamp_duty_sell: float


@dataclass(frozen=True)
class BrokerageConfig:
    delivery_rate: float
    delivery_flat: float
    intraday_rate: float
    intraday_flat_cap: float


@dataclass(frozen=True)
class SlippageConfig:
    """ASSUMPTION, not a measured Indian parameter. See src/costs/india.py."""

    participation_coefficient: float
    max_slippage_fraction: float


@dataclass(frozen=True)
class ImpactConfig:
    """ASSUMPTION. Square-root form is standard; the coefficient is not calibrated to India."""

    coefficient: float
    default_volatility: float


@dataclass(frozen=True)
class RateSchedule:
    """Every statutory and exchange rate in force from ``effective_from`` until the next entry.

    Held as complete snapshots rather than deltas. A schedule where each entry overrides only the
    fields that changed reads compactly and then, three edits later, silently applies a rate nobody
    intended; a full snapshot per epoch is longer and cannot do that.
    """

    effective_from: date
    label: str
    delivery: SegmentRates
    intraday: SegmentRates
    exchange_transaction_charge: float
    ipft_charge: float
    sebi_turnover_fee: float
    gst_rate: float
    brokerage: BrokerageConfig


@dataclass(frozen=True)
class CostsConfig:
    """Indian transaction costs. Statutory rates are sourced and dated in config.yaml.

    Rates are **time-varying**: Union Budgets and SEBI circulars move them, and applying today's
    rates to a 2020 fill is an anachronism that looks perfectly healthy in the output. ``schedule``
    therefore holds one entry per epoch and :meth:`rates_on` selects by trade date.
    """

    verified_on: date
    schedule: tuple[RateSchedule, ...]      # ascending by effective_from, validated at load
    dp_mode: str                            # "retail" (what an investor pays) | "research" (CDSL)
    dp_charge_by_mode: dict[str, float]
    slippage: SlippageConfig
    impact: ImpactConfig

    def rates_on(self, day: date) -> RateSchedule:
        """The schedule entry in force on ``day``.

        Raises rather than falling back to the earliest entry: pricing a trade from before the
        schedule begins would charge it rates that did not exist, which is the precise failure this
        whole structure was introduced to remove.
        """
        applicable = [entry for entry in self.schedule if entry.effective_from <= day]
        if not applicable:
            raise ConfigError(
                f"no cost schedule covers {day}; the earliest entry begins "
                f"{self.schedule[0].effective_from}. Extend costs.schedule with a sourced entry "
                "rather than pricing this trade at rates that were not in force."
            )
        return applicable[-1]

    @property
    def dp_charge_per_scrip_sell(self) -> float:
        """The per-scrip sell-side depository charge under the configured mode."""
        return self.dp_charge_by_mode[self.dp_mode]


@dataclass(frozen=True)
class ConstraintsConfig:
    """Tradability limits. A strategy violating these could not have been executed as backtested."""

    max_participation_rate: float
    min_adv_rupees: float
    adv_window_sessions: int
    circuit_band: float


@dataclass(frozen=True)
class Config:
    meta: MetaConfig
    paths: PathsConfig
    calendar: CalendarConfig
    dates: DatesConfig
    universe: UniverseConfig
    data: DataConfig
    audit: AuditConfig
    costs: CostsConfig
    constraints: ConstraintsConfig
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


def _build_audit(audit: dict[str, Any]) -> AuditConfig:
    """Assemble the audit section. Split out of ``_build`` only to keep that function short."""
    return AuditConfig(
        model_tag=str(_require(audit, "model_tag", "audit")),
        ollama_host=str(_require(audit, "ollama_host", "audit")),
        num_ctx=int(_require(audit, "num_ctx", "audit")),
        request_timeout_seconds=float(_require(audit, "request_timeout_seconds", "audit")),
        probe_timeout_seconds=float(_require(audit, "probe_timeout_seconds", "audit")),
    )


def _segment_rates(section: dict[str, Any], where: str) -> SegmentRates:
    return SegmentRates(
        stt_buy=float(_require(section, "stt_buy", where)),
        stt_sell=float(_require(section, "stt_sell", where)),
        stamp_duty_buy=float(_require(section, "stamp_duty_buy", where)),
        stamp_duty_sell=float(_require(section, "stamp_duty_sell", where)),
    )


def _rate_schedule(entry: dict[str, Any], index: int) -> RateSchedule:
    """One epoch of the statutory schedule."""
    where = f"costs.schedule[{index}]"
    brokerage = _require(entry, "brokerage", where)
    return RateSchedule(
        effective_from=_parse_date(_require(entry, "effective_from", where),
                                   f"{where}.effective_from"),
        label=str(_require(entry, "label", where)),
        delivery=_segment_rates(_require(entry, "delivery", where), f"{where}.delivery"),
        intraday=_segment_rates(_require(entry, "intraday", where), f"{where}.intraday"),
        exchange_transaction_charge=float(
            _require(entry, "exchange_transaction_charge", where)),
        ipft_charge=float(_require(entry, "ipft_charge", where)),
        sebi_turnover_fee=float(_require(entry, "sebi_turnover_fee", where)),
        gst_rate=float(_require(entry, "gst_rate", where)),
        brokerage=BrokerageConfig(
            delivery_rate=float(_require(brokerage, "delivery_rate", f"{where}.brokerage")),
            delivery_flat=float(_require(brokerage, "delivery_flat", f"{where}.brokerage")),
            intraday_rate=float(_require(brokerage, "intraday_rate", f"{where}.brokerage")),
            intraday_flat_cap=float(
                _require(brokerage, "intraday_flat_cap", f"{where}.brokerage")),
        ),
    )


def _build_costs(costs: dict[str, Any]) -> CostsConfig:
    """Assemble the costs section. Every statutory rate here is sourced and dated in config.yaml."""
    slippage = _require(costs, "slippage", "costs")
    impact = _require(costs, "impact", "costs")

    raw_schedule = _require(costs, "schedule", "costs")
    if not isinstance(raw_schedule, list) or not raw_schedule:
        raise ConfigError("costs.schedule must be a non-empty list of rate epochs")
    schedule = tuple(_rate_schedule(entry, i) for i, entry in enumerate(raw_schedule))
    # Ascending order is what rates_on() relies on to pick the last applicable entry. Checked
    # rather than sorted: a schedule written out of order is a mistake worth surfacing, not
    # silently repairing.
    dates = [entry.effective_from for entry in schedule]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ConfigError(
            f"costs.schedule must be in strictly ascending effective_from order, got {dates}"
        )

    dp_charge = _require(costs, "dp_charge", "costs")
    dp_mode = str(_require(costs, "dp_mode", "costs"))
    dp_by_mode = {str(k): float(v) for k, v in dp_charge.items()}
    if dp_mode not in dp_by_mode:
        raise ConfigError(
            f"costs.dp_mode is '{dp_mode}' but costs.dp_charge defines {sorted(dp_by_mode)}"
        )

    return CostsConfig(
        verified_on=_parse_date(_require(costs, "verified_on", "costs"), "costs.verified_on"),
        schedule=schedule,
        dp_mode=dp_mode,
        dp_charge_by_mode=dp_by_mode,
        slippage=SlippageConfig(
            participation_coefficient=float(
                _require(slippage, "participation_coefficient", "costs.slippage")),
            max_slippage_fraction=float(
                _require(slippage, "max_slippage_fraction", "costs.slippage")),
        ),
        impact=ImpactConfig(
            coefficient=float(_require(impact, "coefficient", "costs.impact")),
            default_volatility=float(_require(impact, "default_volatility", "costs.impact")),
        ),
    )


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
            index_symbol=str(_require(cal, "index_symbol", "calendar")),
        ),
        dates=DatesConfig(
            dev_start=_parse_date(_require(dates, "dev_start", "dates"), "dates.dev_start"),
            dev_end=_parse_date(_require(dates, "dev_end", "dates"), "dates.dev_end"),
            holdout_start=_parse_date(_require(dates, "holdout_start", "dates"),
                                      "dates.holdout_start"),
            holdout_end=_parse_date(_require(dates, "holdout_end", "dates"),
                                    "dates.holdout_end"),
        ),
        universe=UniverseConfig(
            method=str(_require(uni, "method", "universe")),
            size=int(_require(uni, "size", "universe")),
            trailing_sessions=int(_require(uni, "trailing_sessions", "universe")),
            rebalance=str(_require(uni, "rebalance", "universe")),
            entry_rank=int(_require(uni, "entry_rank", "universe")),
            exit_rank=int(_require(uni, "exit_rank", "universe")),
            min_listed_sessions=int(_require(uni, "min_listed_sessions", "universe")),
            min_traded_fraction=float(_require(uni, "min_traded_fraction", "universe")),
        ),
        data=DataConfig(prices=PricesConfig(
            authoritative_source=prices.get("authoritative_source"),
            cross_check_rel_tol=float(_require(prices, "cross_check_rel_tol", "data.prices")),
            max_discrepancy_rate=float(_require(prices, "max_discrepancy_rate", "data.prices")),
        )),
        audit=_build_audit(_require(raw, "audit", "root")),
        costs=_build_costs(_require(raw, "costs", "root")),
        constraints=ConstraintsConfig(
            max_participation_rate=float(_require(
                _require(raw, "constraints", "root"), "max_participation_rate", "constraints")),
            min_adv_rupees=float(_require(
                _require(raw, "constraints", "root"), "min_adv_rupees", "constraints")),
            adv_window_sessions=int(_require(
                _require(raw, "constraints", "root"), "adv_window_sessions", "constraints")),
            circuit_band=float(_require(
                _require(raw, "constraints", "root"), "circuit_band", "constraints")),
        ),
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
    if cfg.dates.holdout_end <= cfg.dates.holdout_start:
        raise ConfigError(
            f"holdout_end ({cfg.dates.holdout_end}) must be after "
            f"holdout_start ({cfg.dates.holdout_start})"
        )
    return cfg
