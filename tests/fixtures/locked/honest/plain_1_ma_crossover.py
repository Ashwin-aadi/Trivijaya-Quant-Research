"""A plain trailing moving-average crossover, one of the oldest technical rules there is.

Negative control: every average is computed from ``view.closes(long_window)``, itself already
truncated to sessions strictly before the decision date. The short average is a trailing tail of
that same bounded window, never a centred or forward-looking window.
"""

from __future__ import annotations

from datetime import date

from src.backtest.strategy import MarketView, Signal, Strategy


def _latest_visible_session(view: MarketView) -> date:
    """The newest session already printed to the tape, the only date a signal may cite."""
    recent = view.history(1)
    if recent.is_empty():
        return date.min
    latest = recent["session_date"].max()
    assert isinstance(latest, date)
    return latest


class MovingAverageCrossover(Strategy):
    """Hold names whose short trailing average sits above their long trailing average."""

    rationale = (
        "A short trailing average rising above a longer trailing average is the textbook "
        "trend-following signal: recent price action is running ahead of the slower average, "
        "which is read as an uptrend. It is included as a plain, probably-marginal baseline, "
        "not a claim of edge."
    )

    def __init__(self, short_window: int = 20, long_window: int = 100, top_n: int = 5) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        decision_date = _latest_visible_session(view)
        closes = view.closes(self.long_window)
        if closes.height < self.long_window:
            return Signal(information_available_at=decision_date, weights={})
        gaps: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            series = closes[symbol]
            if series.null_count() > 0:
                continue
            # tail() on an already-trailing window is still trailing: both averages end at the
            # same, most recent visible session and start no later than it.
            short_avg = series.tail(self.short_window).mean()
            long_avg = series.mean()
            if short_avg is None or long_avg is None or long_avg == 0:
                continue
            gaps[symbol] = (short_avg - long_avg) / long_avg
        ranked = sorted(
            (item for item in gaps.items() if item[1] > 0), key=lambda kv: kv[1], reverse=True
        )[: self.top_n]
        weights: dict[str, float] = {}
        if ranked:
            share = 1.0 / len(ranked)
            for symbol, _gap in ranked:
                weights[symbol] = share
        return Signal(information_available_at=decision_date, weights=weights)
