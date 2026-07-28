"""Honest trailing trend-following strategy that happens to use the words 'final' and 'end'.

Negative-control fixture: ``end_state`` names the most recent close already visible in
``view`` -- the end of the *visible* window, not any later point in time. ``final_allocation``
names the weight dict this strategy settles on for the upcoming session, which is precisely what
every ``Signal.weights`` dict is. Nothing here reaches past the last session ``view`` reveals.
"""

from __future__ import annotations

from datetime import date

from src.backtest.strategy import MarketView, Signal, Strategy


class TrapFinalAndEnd(Strategy):
    """Hold names whose latest visible close sits above their trailing average."""

    rationale = (
        "A price sitting above its own trailing average is the standard, simple definition of "
        "an uptrend: recent demand has outpaced the level buyers were willing to pay over the "
        "look-back window. This strategy leans into names in that state and holds nothing when "
        "none qualify. It is a textbook trend filter, offered as a plain baseline rather than a "
        "claim of edge."
    )

    def __init__(self, trend_window: int = 50, target_gross: float = 1.0) -> None:
        self.trend_window = trend_window
        self.target_gross = target_gross

    @staticmethod
    def _last_session(view: MarketView) -> date:
        recent = view.history(1)
        if recent.is_empty():
            return date.min
        latest = recent["session_date"].max()
        assert isinstance(latest, date)
        return latest

    def generate(self, view: MarketView) -> Signal:
        decision_date = self._last_session(view)
        closes = view.closes(self.trend_window)
        if closes.height < self.trend_window:
            return Signal(information_available_at=decision_date, weights={})
        # "end_state" is the state of the trailing window at its own end -- the newest close
        # already printed -- not a state reached at the end of some later, unseen period.
        end_state = closes.tail(1).row(0, named=True)
        above_trend: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            series = closes[symbol]
            if series.null_count() > 0:
                continue
            trailing_mean = series.mean()
            current = end_state.get(symbol)
            if current is None or trailing_mean is None:
                continue
            if current > trailing_mean:
                above_trend.append(symbol)
        final_allocation: dict[str, float] = {}
        if above_trend:
            share = self.target_gross / len(above_trend)
            for symbol in above_trend:
                final_allocation[symbol] = share
        return Signal(information_available_at=decision_date, weights=final_allocation)
