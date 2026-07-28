"""Honest momentum-with-data-screen strategy using the words 'survivor' and 'forward'.

Negative-control fixture: ``survivor_count`` counts symbols, within the point-in-time universe
the engine already handed this strategy via ``view.symbols``, that have enough trailing history
to be scored -- a data-sufficiency screen, not a selection made using end-of-period index
membership. ``forward_weight`` is simply the weight carried into the next session, exactly what
``Signal.weights`` always represents.
"""

from __future__ import annotations

from datetime import date

from src.backtest.strategy import MarketView, Signal, Strategy


class TrapSurvivorAndForward(Strategy):
    """Rank symbols with sufficient trailing history by momentum and go long the leaders."""

    rationale = (
        "Trailing return over a few months is a standard, unglamorous momentum proxy. Requiring "
        "a minimum number of trailing observations before scoring a name is an honest "
        "data-quality screen -- thinly-traded or recently-added names are dropped rather than "
        "scored on too little history -- and is unrelated to which names later remained in any "
        "index."
    )

    def __init__(self, lookback: int = 40, min_history: int = 30, top_n: int = 4) -> None:
        self.lookback = lookback
        self.min_history = min_history
        self.top_n = top_n

    @staticmethod
    def _decision_date(view: MarketView) -> date:
        recent = view.history(1)
        if recent.is_empty():
            return date.min
        latest = recent["session_date"].max()
        assert isinstance(latest, date)
        return latest

    def generate(self, view: MarketView) -> Signal:
        decision_date = self._decision_date(view)
        closes = view.closes(self.lookback)
        eligible: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            observed = closes[symbol].drop_nulls()
            if observed.len() >= self.min_history:
                eligible.append(symbol)
        # "survivor_count" -- how many symbols in *today's* point-in-time universe have enough
        # trailing history to be scored. It says nothing about which symbols exist at any later
        # date; the universe itself came from view.symbols, already point-in-time.
        survivor_count = len(eligible)
        if survivor_count == 0:
            return Signal(information_available_at=decision_date, weights={})
        first_row = closes.head(1).row(0, named=True)
        last_row = closes.tail(1).row(0, named=True)
        scores: dict[str, float] = {}
        for symbol in eligible:
            start = first_row.get(symbol)
            end = last_row.get(symbol)
            if start is None or end is None or start <= 0:
                continue
            scores[symbol] = (end / start) - 1.0
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.top_n]
        forward_weight: dict[str, float] = {}
        if ranked:
            share = 1.0 / len(ranked)
            for symbol, _score in ranked:
                forward_weight[symbol] = share
        return Signal(information_available_at=decision_date, weights=forward_weight)
