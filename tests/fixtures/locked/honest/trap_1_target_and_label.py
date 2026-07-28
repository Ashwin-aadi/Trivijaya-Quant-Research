"""Honest trailing-momentum ranking strategy that happens to use the words 'target' and 'label'.

Negative-control fixture: every number this strategy touches comes from ``view``, which the
engine has already truncated to sessions strictly before the decision date. The variable named
``label`` is nothing but a per-symbol momentum score computed from trailing prices — not a
supervised-learning target smuggled in from the future. The variable named ``target_weight`` is
simply the portfolio weight this strategy targets going forward, which is exactly what
``Signal.weights`` always means.
"""

from __future__ import annotations

from datetime import date

from src.backtest.strategy import MarketView, Signal, Strategy


def _decision_stamp(view: MarketView) -> date:
    """The newest session already printed to the tape, the only date a signal may cite."""
    recent = view.history(1)
    if recent.is_empty():
        return date.min
    latest = recent["session_date"].max()
    assert isinstance(latest, date)
    return latest


class TrapTargetAndLabel(Strategy):
    """Rank symbols by trailing return and hold the top scorers equally weighted."""

    rationale = (
        "Recent trailing return is a simple, well-documented proxy for continuing momentum over "
        "horizons of a few months in equity markets. Ranking the visible universe by that "
        "trailing score and holding the strongest performers equally weighted is a plain "
        "momentum tilt, not a claim of any special edge. Every input to the score is a price "
        "that had already printed before the decision was made."
    )

    def __init__(self, lookback: int = 63, top_n: int = 3) -> None:
        self.lookback = lookback
        self.top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        decision_date = _decision_stamp(view)
        closes = view.closes(self.lookback)
        if closes.height < self.lookback:
            return Signal(information_available_at=decision_date, weights={})
        first_row = closes.head(1).row(0, named=True)
        last_row = closes.tail(1).row(0, named=True)
        label: dict[str, float] = {}
        for symbol in view.symbols:
            start = first_row.get(symbol)
            end = last_row.get(symbol)
            if start is None or end is None or start <= 0:
                continue
            # "label" here is just the score being ranked on, computed entirely from the
            # trailing window above -- it is never the return of a session not yet visible.
            label[symbol] = (end / start) - 1.0
        ranked = sorted(label.items(), key=lambda kv: kv[1], reverse=True)[: self.top_n]
        target_weight: dict[str, float] = {}
        if ranked:
            share = 1.0 / len(ranked)
            for symbol, _score in ranked:
                target_weight[symbol] = share
        return Signal(information_available_at=decision_date, weights=target_weight)
