"""Hold the universe only on sessions that follow certain weekdays, and hold cash otherwise."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible

# Monday and Tuesday, using date.weekday()'s numbering. The choice is arbitrary in the sense that
# no mechanism recommends these two days over any others.
DEFAULT_WEEKDAYS: tuple[int, ...] = (0, 1)


class SeasonalityDayOfWeek(Strategy):
    """A deliberately weak calendar rule, included to measure what a null looks like."""

    rationale = (
        "Day-of-week effects are among the most-tested and least-durable regularities in equity "
        "returns, and any that survive are usually too small to clear costs. This holds the whole "
        "universe when the most recent completed session fell on a Monday or a Tuesday and sits "
        "in cash otherwise. There is no mechanism behind it beyond the historical record, so it "
        "is expected to fail; it is here as a weak-signal reference point."
    )

    def __init__(self, weekdays: tuple[int, ...] = DEFAULT_WEEKDAYS) -> None:
        self._weekdays = weekdays

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        if not view.symbols or view.history(lookback=1).is_empty():
            return Signal(information_available_at=stamp, weights={})

        # The weekday tested is that of the last completed session, so the position is carried
        # over the session that follows it. Reading the traded session's own calendar date would
        # mean reaching past the decision moment, which this class never does.
        if stamp.weekday() not in self._weekdays:
            return Signal(information_available_at=stamp, weights={})
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(sorted(view.symbols)),
        )
