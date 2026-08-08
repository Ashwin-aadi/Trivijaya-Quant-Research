from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines a breakout signal with a momentum indicator to select stocks "
        "that have both recently broken out and show strong relative strength."
    )

    def __init__(self, window_breakout: int = 20, top_n_breakout: int = 5, lookback_momentum: int = 60) -> None:
        self._window_breakout = window_breakout
        self._top_n_breakout = top_n_breakout
        self._lookback_momentum = lookback_momentum

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_breakout)
        if closes.height < self._window_breakout:
            return Signal(information_available_at=stamp, weights={})

        breakout_picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_breakout:
                continue
            if values[-1] >= max(values):
                breakout_picks.append(symbol)

        breakout_picks = breakout_picks[:self._top_n_breakout]
        if not breakout_picks:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        history = view.history(lookback=self._lookback_momentum)
        for symbol in breakout_picks:
            recent_closes = [float(v) for v in history["close"][symbol].drop_nulls().to_list()]
            if len(recent_closes) < self._lookback_momentum:
                continue
            momentum_score = sum(recent_closes[-10:]) / 10.0
            momentum_scores[symbol] = momentum_score

        top_picks: list[str] = []
        for symbol, score in sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True):
            if len(top_picks) >= self._top_n_breakout:
                break
            top_picks.append(symbol)

        weight = 1.0 / len(top_picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest