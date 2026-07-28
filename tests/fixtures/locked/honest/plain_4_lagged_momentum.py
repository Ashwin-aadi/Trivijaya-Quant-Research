"""A plain momentum rule with a deliberate positive lag between formation and use.

Negative control: the daily return series is built with ``shift(1)`` -- comparing each session to
the one immediately before it -- and the most recent ``skip`` observations are then dropped from
the *end* of that trailing series, moving the score further into the past rather than closer to
the future. Both operations only ever restrict attention to older data.
"""

from __future__ import annotations

from datetime import date

from src.backtest.strategy import MarketView, Signal, Strategy


class LaggedMomentumStrategy(Strategy):
    """Rank names by trailing return, excluding the most recent sessions from the score."""

    rationale = (
        "Momentum formed on returns further in the past, deliberately excluding the most recent "
        "few sessions, targets the classic academic momentum anomaly while side-stepping the "
        "short-horizon reversal effect documented in the same literature. Every return entering "
        "the score is still strictly historical -- only how recent, within the trailing window, "
        "it is allowed to be is restricted."
    )

    def __init__(self, formation_window: int = 126, skip: int = 5, top_n: int = 5) -> None:
        if skip < 1 or formation_window - skip < 2:
            raise ValueError("skip must leave at least two trailing returns to score on")
        self.formation_window = formation_window
        self.skip = skip
        self.top_n = top_n

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
        closes = view.closes(self.formation_window)
        if closes.height < self.formation_window:
            return Signal(information_available_at=decision_date, weights={})
        scored: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            price = closes[symbol]
            if price.null_count() > 0:
                continue
            # daily_return[i] compares session i to session i-1: a one-step trailing lag.
            daily_return = price / price.shift(1) - 1.0
            # Drop the first null (from the shift) and the most recent `skip` sessions -- both
            # trims move the score away from the boundary, never toward it.
            usable = daily_return[1 : daily_return.len() - self.skip]
            if usable.is_empty():
                continue
            score = usable.mean()
            if score is not None:
                scored[symbol] = float(score)
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[: self.top_n]
        weights: dict[str, float] = {}
        if ranked:
            share = 1.0 / len(ranked)
            for symbol, _score in ranked:
                weights[symbol] = share
        return Signal(information_available_at=decision_date, weights=weights)
