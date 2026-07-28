"""Category ``full_sample_fit``, variant ``obvious``.

Defect: a standardising transform is fitted once, in the constructor, over every session in the
price frame — including every session that is still in the future at each decision point the
strategy will later face. The per-symbol location and scale therefore encode where each stock's
price eventually settles, and a z-score computed against them means "cheap relative to the whole
period", not "cheap relative to what was known at the time".

The transform is hand-rolled rather than imported so the fixture has no third-party dependency, but
it follows the ``fit`` then ``transform`` idiom exactly, and is fitted in the place the mistake is
usually made: before anything has been split.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class StandardScaler:
    """Per-symbol location and scale, in the usual fit-once-transform-many idiom."""

    def __init__(self) -> None:
        self._stats: dict[str, tuple[float, float]] = {}

    def fit(self, frame: pl.DataFrame) -> StandardScaler:
        """Learn a mean and a standard deviation per symbol from ``frame``."""
        summary = frame.group_by("symbol").agg(
            pl.col("adj_close").mean().alias("centre"),
            pl.col("adj_close").std().alias("spread"),
        )
        self._stats = {
            str(row["symbol"]): (float(row["centre"] or 0.0), float(row["spread"] or 1.0) or 1.0)
            for row in summary.iter_rows(named=True)
        }
        return self

    def transform(self, symbol: str, value: float) -> float:
        """Standardise one observation with the statistics learned in :meth:`fit`."""
        centre, spread = self._stats.get(symbol, (0.0, 1.0))
        return (value - centre) / spread


class FullSampleFitObvious(Strategy):
    """Buys the names standing furthest below their standardised level."""

    rationale = (
        "Prices are standardised per name so that a two hundred rupee stock and a two thousand "
        "rupee stock can be compared on one scale. The book then holds the names trading furthest "
        "below their own typical level, on the expectation that price levels are mean-reverting "
        "over horizons of a few months."
    )

    def __init__(self, panel: pl.DataFrame, top_n: int = 10) -> None:
        self._top_n = top_n
        # THE CHEAT: fitted here, over the entire frame, before any train/test boundary exists.
        # Every statistic the scaler holds summarises data that postdates most of the decisions it
        # will be used for.
        self._scaler = StandardScaler().fit(panel)

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        latest = view.latest_close()
        if not latest:
            return Signal(information_available_at=stamp, weights={})

        scores = {symbol: self._scaler.transform(symbol, price) for symbol, price in latest.items()}
        chosen = sorted(scores, key=lambda s: (scores[s], s))[: self._top_n]
        return Signal(information_available_at=stamp, weights=_spread(chosen))


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
