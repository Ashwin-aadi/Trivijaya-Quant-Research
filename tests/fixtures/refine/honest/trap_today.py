"""Hold the whole universe except when the most recent visible session fell on a Friday.

Naming note: the local ``_today`` is the last session the view will serve. It is the strategy's
own "today" and is strictly earlier than the session being traded. It is deliberately not
``view.as_of``: that is the fill date, and stamping a signal with it would claim knowledge the
strategy could not have had.
"""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible

# date.weekday() numbering, so 4 is Friday. Skipping it means holding cash over the weekend gap.
DEFAULT_SKIP_WEEKDAYS: tuple[int, ...] = (4,)


class TodayWeekdayFilter(Strategy):
    """A calendar rule keyed on the weekday of the last completed session."""

    rationale = (
        "Holding cash over the weekend removes the gap risk of two non-trading days during which "
        "news can accumulate with no chance to react. That is a real reduction in exposure, not a "
        "source of return, and the historical evidence for a weekend effect in returns is weak "
        "and heavily data-mined. This is expected to underperform simply holding the universe, "
        "and is included as a reference point for what a weak calendar rule looks like."
    )

    def __init__(self, skip_weekdays: tuple[int, ...] = DEFAULT_SKIP_WEEKDAYS) -> None:
        if any(day < 0 or day > 6 for day in skip_weekdays):
            raise ValueError("weekdays must lie in [0, 6]")
        self._skip = skip_weekdays

    def generate(self, view: MarketView) -> Signal:
        _today = latest_visible(view)
        if not view.symbols or view.history(lookback=1).is_empty():
            return Signal(information_available_at=_today, weights={})

        # The weekday tested belongs to the last completed session. The traded session's own date
        # is never consulted, because that would be reaching past the decision moment.
        if _today.weekday() in self._skip:
            return Signal(information_available_at=_today, weights={})
        return Signal(
            information_available_at=_today,
            weights=equal_weight(sorted(view.symbols)),
        )
