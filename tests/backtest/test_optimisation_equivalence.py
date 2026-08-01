"""Prove the optimised MarketView returns exactly what the approved P1 implementation returned.

This file discharges the PI's ruling of 2026-08-01:

    "No P2 result may be generated from an engine whose equivalence to the approved P1
    implementation has not been demonstrated."

Two levels of check, because they fail differently:

* **Accessor-level** — every frame ``MarketView`` hands a strategy is compared cell-for-cell
  against :class:`ReferenceMarketView`, across many dates and awkward symbol sets. This catches a
  wrong slice directly, at the point of the mistake.
* **Run-level** — full backtests are run through both implementations and the entire per-session
  record is compared: equity, net return, gross return, exposure, turnover and cost. Comparing
  only the final Sharpe would let compensating errors cancel — a position wrong one way on Tuesday
  and the other way on Wednesday nets out in a summary statistic and not in the series.

The strategy set is chosen for **code-path coverage, not size**: leaky fixtures (extreme values
stress the arithmetic), factors (varied turnover and holding periods), and a seeded sample of
generated survivors (unpredictable access patterns written by a model, not by us).
"""

from __future__ import annotations

import importlib
import importlib.util
import random
from datetime import date
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from src.backtest._reference import ReferenceMarketView
from src.backtest.engine import BacktestEngine
from src.backtest.strategy import MarketView, PanelIndex, Strategy
from src.common.config import load_config
from src.costs.india import CostModel
from src.data.calendar import load_calendar

SEED = 42
SURVIVORS = Path("benchmarks/alphaaudit/survivors")

#: Factors, chosen for spread of turnover and holding period.
FACTOR_FIXTURES = (
    "equal_weight_universe",
    "momentum_skip_month",
    "mean_reversion_5d",
    "low_volatility",
    "inverse_volatility_weighted",
    "long_term_reversal_756d",
    "relative_strength_vs_universe",
    "bollinger_reversion",
    "high_volatility",
    "dual_momentum_21_126",
    "random_walk_baseline",
)

#: Deliberate cheats. Their extreme values are the harshest arithmetic the engine ever sees.
LEAKY_FIXTURES = ("leak_future_return", "leak_full_sample_scaler", "leak_survivorship")


def _data_available() -> bool:
    cfg = load_config()
    return (cfg.paths.data_processed / "prices_adjusted.parquet").exists()


pytestmark = pytest.mark.skipif(
    not _data_available(),
    reason="requires the built price panel; equivalence is checked where the data lives",
)


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    cfg = load_config()
    return pl.read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")


@pytest.fixture(scope="module")
def universe() -> pl.DataFrame:
    cfg = load_config()
    return pl.read_parquet(cfg.paths.data_processed / "universe.parquet")


def _load(module_path: str) -> type[Strategy]:
    """Import a strategy by dotted path, or by file path for the survivors corpus.

    The survivors directory is deliberately not a package — it is frozen data, and adding an
    __init__.py would modify the artefact the published leak counts describe. So it is loaded
    from file, the same way scripts/run_corpus_backtest.py loads it.
    """
    if module_path.startswith("benchmarks."):
        path = SURVIVORS / f"{module_path.rsplit('.', 1)[1]}.py"
        spec = importlib.util.spec_from_file_location(f"equiv_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    found = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, Strategy) and value is not Strategy
    ]
    return found[-1]


# --- accessor-level equivalence -------------------------------------------------


@pytest.mark.parametrize(
    "as_of",
    [date(2020, 3, 2), date(2020, 6, 15), date(2021, 11, 29), date(2023, 4, 13),
     date(2024, 12, 31)],
)
@pytest.mark.parametrize("lookback", [None, 1, 5, 21, 63, 126, 252, 756])
def test_history_matches_the_reference(panel: pl.DataFrame, universe: pl.DataFrame,
                                       as_of: date, lookback: int | None) -> None:
    symbols = tuple(
        universe.filter(pl.col("rebalance_date") <= as_of)
        .filter(pl.col("rebalance_date") == pl.col("rebalance_date").max())["symbol"]
        .to_list()
    )
    index = PanelIndex(panel)
    fast = MarketView(panel, as_of=as_of, symbols=symbols, index=index).history(lookback)
    slow = ReferenceMarketView(panel, as_of=as_of, symbols=symbols).history(lookback)
    assert fast.equals(slow), (
        f"history(lookback={lookback}) at {as_of} differs: "
        f"{fast.height} rows vs {slow.height}"
    )


@pytest.mark.parametrize("as_of", [date(2020, 3, 2), date(2022, 7, 11), date(2024, 10, 1)])
@pytest.mark.parametrize("lookback", [None, 5, 21, 252])
def test_closes_matches_the_reference(panel: pl.DataFrame, universe: pl.DataFrame,
                                      as_of: date, lookback: int | None) -> None:
    symbols = tuple(
        universe.filter(pl.col("rebalance_date") <= as_of)
        .filter(pl.col("rebalance_date") == pl.col("rebalance_date").max())["symbol"]
        .to_list()
    )
    fast = MarketView(panel, as_of=as_of, symbols=symbols).closes(lookback)
    slow = ReferenceMarketView(panel, as_of=as_of, symbols=symbols).closes(lookback)
    assert fast.equals(slow)


@pytest.mark.parametrize("as_of", [date(2020, 3, 2), date(2023, 4, 13)])
def test_latest_close_matches_the_reference(panel: pl.DataFrame, universe: pl.DataFrame,
                                            as_of: date) -> None:
    symbols = tuple(
        universe.filter(pl.col("rebalance_date") <= as_of)
        .filter(pl.col("rebalance_date") == pl.col("rebalance_date").max())["symbol"]
        .to_list()
    )
    fast = MarketView(panel, as_of=as_of, symbols=symbols).latest_close()
    slow = ReferenceMarketView(panel, as_of=as_of, symbols=symbols).latest_close()
    assert fast == slow


def test_awkward_symbol_sets_still_match(panel: pl.DataFrame) -> None:
    """Sparse and absent symbols are where a session-slicing shortcut is most likely to differ.

    A symbol that trades rarely means the last N *calendar* sessions and the last N sessions
    *this symbol appears in* are different sets. The optimised path widens its slice until enough
    qualifying sessions are found; this asserts that widening lands on the reference's answer.
    """
    as_of = date(2023, 6, 1)
    all_symbols = panel["symbol"].unique().to_list()
    rng = random.Random(SEED)
    cases: list[tuple[str, ...]] = [
        (),                                                    # empty universe
        ("RELIANCE",),                                         # single liquid name
        ("__NOT_A_SYMBOL__",),                                 # absent entirely
        ("RELIANCE", "__NOT_A_SYMBOL__"),                      # mixed
        tuple(rng.sample(all_symbols, 3)),                     # arbitrary thin set
        tuple(rng.sample(all_symbols, 40)),                    # arbitrary wide set
    ]
    for symbols in cases:
        for lookback in (None, 1, 21, 252):
            fast = MarketView(panel, as_of=as_of, symbols=symbols).history(lookback)
            slow = ReferenceMarketView(panel, as_of=as_of, symbols=symbols).history(lookback)
            assert fast.equals(slow), f"symbols={symbols} lookback={lookback}"


def test_panel_index_rejects_an_unsorted_panel(panel: pl.DataFrame) -> None:
    """A silently unsorted panel would make every slice wrong while still returning numbers."""
    shuffled = panel.head(50_000).sample(fraction=1.0, shuffle=True, seed=SEED)
    with pytest.raises(ValueError, match="sorted by session_date"):
        PanelIndex(shuffled)


# --- run-level equivalence ------------------------------------------------------


class _ReferenceEngine(BacktestEngine):
    """The engine with the original view swapped back in, for whole-run comparison.

    ``ReferenceMarketView`` is not a subclass of ``MarketView`` and must not become one - it is a
    frozen fossil, and inheriting from the thing it exists to check would make the check circular.
    It implements the same accessors, which is all the engine uses, so the substitution is
    duck-typed and the cast says exactly that.
    """

    def _make_view(self, as_of: date, symbols: tuple[str, ...]) -> MarketView:
        reference = ReferenceMarketView(self._panel, as_of=as_of, symbols=symbols)
        return cast(MarketView, reference)


def _survivor_sample(n: int) -> list[str]:
    names = sorted(p.stem for p in SURVIVORS.glob("candidate_*.py"))
    return random.Random(SEED).sample(names, min(n, len(names)))


def _instantiate(strategy_class: type[Strategy], panel: pl.DataFrame,
                 universe: pl.DataFrame) -> Strategy:
    """Construct a strategy, supplying whichever dependency its constructor demands.

    The leaky fixtures deliberately take the full panel or the end-of-period universe — that
    IS their cheat, and it is why they are the harshest test of the arithmetic. Their
    signatures are part of the frozen artefact and are not changed to suit this test.
    """
    import inspect

    parameters = inspect.signature(strategy_class.__init__).parameters
    if "panel" in parameters:
        return strategy_class(panel)      # type: ignore[call-arg]
    if "universe" in parameters:
        return strategy_class(universe)   # type: ignore[call-arg]
    return strategy_class()


def _compare_runs(panel: pl.DataFrame, universe: pl.DataFrame, module_path: str) -> None:
    cfg = load_config()
    calendar = load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet")
    strategy_class = _load(module_path)

    fast_engine = BacktestEngine(
        panel=panel, calendar=calendar, universe=universe, cost_model=CostModel(cfg.costs)
    )
    slow_engine = _ReferenceEngine(
        panel=panel, calendar=calendar, universe=universe, cost_model=CostModel(cfg.costs)
    )
    fast = fast_engine.run(
        _instantiate(strategy_class, panel, universe),
        start=cfg.dates.dev_start, end=cfg.dates.dev_end,
    )
    slow = slow_engine.run(
        _instantiate(strategy_class, panel, universe),
        start=cfg.dates.dev_start, end=cfg.dates.dev_end,
    )

    assert fast.ruined_on == slow.ruined_on, module_path
    assert fast.to_frame().equals(slow.to_frame()), (
        f"{module_path}: per-session record differs between implementations. "
        "Per the PI ruling this is a correctness bug and must be resolved before any P2 run."
    )


@pytest.mark.parametrize("name", LEAKY_FIXTURES)
def test_leaky_fixture_runs_are_identical(panel: pl.DataFrame, universe: pl.DataFrame,
                                          name: str) -> None:
    _compare_runs(panel, universe, f"tests.fixtures.leaky.{name}")


@pytest.mark.parametrize("name", FACTOR_FIXTURES)
def test_factor_runs_are_identical(panel: pl.DataFrame, universe: pl.DataFrame, name: str) -> None:
    _compare_runs(panel, universe, f"tests.fixtures.clean.{name}")


@pytest.mark.parametrize("name", _survivor_sample(8))
def test_generated_survivor_runs_are_identical(panel: pl.DataFrame, universe: pl.DataFrame,
                                               name: str) -> None:
    """Model-written strategies access the view in ways we did not design for. That is the point."""
    _compare_runs(panel, universe, f"benchmarks.alphaaudit.survivors.{name}")
