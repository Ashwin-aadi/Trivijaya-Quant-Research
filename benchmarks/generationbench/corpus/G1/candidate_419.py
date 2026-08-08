from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: a 20-day closing price "
        "breakout and the volume trend. A breakout indicates strong momentum, while rising "
        "volume suggests increased interest in the stock. Together, they can signal a potential"
        " continuation of the upward trend."
    )

    def __init__(self, window_breakout: int = 20, top_n_breakout: int = 5) -> None:
        self._window_breakout = window_breakout
        self._top_n_breakout = top_n_breakout

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_breakout)
        if history.height < self._window_breakout:
            return Signal(information_available_at=stamp, weights={})

        breakout_picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].select("close").drop_nulls().to_list()]
            if len(values) < self._window_breakout:
                continue
            if values[-1] >= max(values):
                breakout_picks.append(symbol)

        breakout_picks = breakout_picks[: self._top_n_breakout]
        if not breakout_picks:
            return Signal(information_available_at=stamp, weights={})

        volume_trend_picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].select("volume").drop_nulls().to_list()]
            if len(values) < self._window_breakout - 1:
                continue
            volume_trend = all(
                values[i] > values[i + 1] for i in range(len(values) - 1)
            )
            if volume_trend:
                volume_trend_picks.append(symbol)

        combined_picks: list[str] = []
        for symbol in breakout_picks:
            if symbol in volume_trend_picks:
                combined_picks.append(symbol)

        if not combined_picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(combined_picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in combined_picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest