"""Honest short-term reversal strategy that happens to use the words 'today' and 'latest'.

Negative-control fixture: ``today`` is bound to the newest session already printed in
``view.history()`` -- never to ``view.as_of``, which is the session the engine is about to fill
and has deliberately not been made visible to this strategy. ``latest_level`` is simply the most
recent close in that same, already-past window. Both names describe "the most recent thing we
are allowed to know," not "the thing that hasn't happened yet."
"""

from __future__ import annotations

from datetime import date

from src.backtest.strategy import MarketView, Signal, Strategy


class TrapTodayAndLatest(Strategy):
    """Buy the recent laggards over a short trailing window, a plain reversal bet."""

    rationale = (
        "Over short horizons of a week or two, equity returns show mild mean reversion: names "
        "that fell hardest recently tend to bounce, and names that ran up hardest tend to cool. "
        "This strategy buys the weakest recent performers in the visible universe as a plain "
        "short-horizon reversal bet, with no claim that the effect survives trading costs."
    )

    def __init__(self, reversal_window: int = 5, top_n: int = 4) -> None:
        self.reversal_window = reversal_window
        self.top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        recent_session = view.history(1)
        # "today", from the strategy's point of view, is the newest session already on the
        # tape -- the session the engine filled last, not the one it is about to fill next.
        today = date.min if recent_session.is_empty() else recent_session["session_date"].max()
        assert isinstance(today, date)
        closes = view.closes(self.reversal_window + 1)
        if closes.height < self.reversal_window + 1:
            return Signal(information_available_at=today, weights={})
        oldest_row = closes.head(1).row(0, named=True)
        latest_level = closes.tail(1).row(0, named=True)
        laggards: dict[str, float] = {}
        for symbol in view.symbols:
            start = oldest_row.get(symbol)
            end = latest_level.get(symbol)
            if start is None or end is None or start <= 0:
                continue
            laggards[symbol] = (end / start) - 1.0
        losers = sorted(laggards.items(), key=lambda kv: kv[1])[: self.top_n]
        weights: dict[str, float] = {}
        if losers:
            share = 1.0 / len(losers)
            for symbol, _ret in losers:
                weights[symbol] = share
        return Signal(information_available_at=today, weights=weights)
