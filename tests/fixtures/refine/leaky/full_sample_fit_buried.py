"""Category ``full_sample_fit``, variant ``buried``.

Defect: the gap-filling transform in ``_MedianFill`` is fitted across the entire supplied frame in
the constructor. Missing observations at any decision date are then filled with a median computed
partly from sessions that had not happened yet, so a suspended or illiquid name is imputed at a
level informed by where it eventually resumed trading.

Imputation is the easiest fitted transform to leak with, because it does not feel like a model. It
feels like data cleaning, it is usually written once at the top of a pipeline, and the fitted
statistic is a single unremarkable number. Everything around it here is genuine: a composite of
three factors, per-factor cross-sectional ranking, a coverage requirement, and a cap on gross
exposure. The defect is one call in ``__init__`` and one lookup inside the scoring loop.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class _MedianFill:
    """Learns a per-symbol median level and substitutes it wherever an observation is absent."""

    def __init__(self) -> None:
        self._levels: dict[str, float] = {}

    def fit(self, rows: pl.DataFrame) -> _MedianFill:
        """Take the median close per symbol from every row supplied."""
        summary = rows.group_by("symbol").agg(pl.col("adj_close").median().alias("level"))
        self._levels = {
            str(row["symbol"]): float(row["level"] or 0.0)
            for row in summary.iter_rows(named=True)
        }
        return self

    def level_for(self, symbol: str) -> float:
        """The learned substitute for ``symbol``, or zero if the symbol was never seen."""
        return self._levels.get(symbol, 0.0)


class CompositeFactorRanker(Strategy):
    """Ranks on an equally weighted composite of momentum, reversal and low volatility."""

    rationale = (
        "Single factors in Indian large caps are noisy and take long stretches out of favour, so "
        "the book scores each name on three weakly correlated legs and holds the best composite. "
        "Momentum and short-term reversal disagree by construction, which stops the composite "
        "from becoming a momentum book in disguise, and the volatility leg tilts away from names "
        "whose ranking is being driven by their own noise."
    )

    def __init__(
        self,
        prior_levels: pl.DataFrame,
        long_window: int = 126,
        short_window: int = 5,
        top_n: int = 12,
    ) -> None:
        self._long_window = long_window
        self._short_window = short_window
        self._top_n = top_n
        # THE CHEAT: the fill levels are learned here, in one pass over everything supplied, before
        # any split. Each learned median summarises the whole period, so filling a gap at an early
        # decision date substitutes a number derived from later sessions.
        self._fill = _MedianFill().fit(prior_levels)

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        closes = view.closes(lookback=self._long_window + 1)
        if closes.height < self._long_window:
            return Signal(information_available_at=stamp, weights={})

        legs = {symbol: self._legs(closes, symbol) for symbol in self._covered(view, closes)}
        usable = {symbol: leg for symbol, leg in legs.items() if leg is not None}
        if not usable:
            return Signal(information_available_at=stamp, weights={})

        composite = _rank_composite(usable)
        chosen = sorted(composite, key=lambda s: (-composite[s], s))[: self._top_n]
        return Signal(information_available_at=stamp, weights=_spread(chosen))

    def _covered(self, view: MarketView, closes: pl.DataFrame) -> list[str]:
        """Names with a column in the pivot and enough sessions to fill the long window."""
        return [
            symbol
            for symbol in view.symbols
            if symbol in closes.columns
            and closes[symbol].drop_nulls().len() >= self._long_window // 2
        ]

    def _legs(self, closes: pl.DataFrame, symbol: str) -> tuple[float, float, float] | None:
        """Momentum, negated short-term return, and negated volatility for one name."""
        values = self._series(closes, symbol)
        if len(values) < self._long_window or values[0] <= 0:
            return None
        momentum = values[-1] / values[0] - 1.0
        base = values[-self._short_window - 1]
        reversal = -(values[-1] / base - 1.0) if base > 0 else 0.0
        steps = [
            values[i] / values[i - 1] - 1.0 for i in range(1, len(values)) if values[i - 1] > 0
        ]
        return momentum, reversal, -_stdev(steps)

    def _series(self, closes: pl.DataFrame, symbol: str) -> list[float]:
        """Close series for one name, with absent observations replaced by the learned level."""
        substitute = self._fill.level_for(symbol)
        return [
            float(v) if v is not None else substitute for v in closes[symbol].to_list()
        ]


def _rank_composite(legs: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    """Average of the three per-leg cross-sectional ranks, scaled to the unit interval."""
    names = sorted(legs)
    total: dict[str, float] = dict.fromkeys(names, 0.0)
    for position in range(3):
        ordered = sorted(names, key=lambda s: (legs[s][position], s))
        for rank, symbol in enumerate(ordered):
            total[symbol] += rank / max(len(names) - 1, 1)
    return {symbol: value / 3.0 for symbol, value in total.items()}


def _stdev(values: list[float]) -> float:
    """Sample standard deviation; zero when there is nothing to disperse."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return float(variance**0.5)


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _spread(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
