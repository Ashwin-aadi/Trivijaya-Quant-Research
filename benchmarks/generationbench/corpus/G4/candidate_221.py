from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy exploits the continuation phase of a breakout by identifying stocks "
        "that have shown strong momentum after breaking out of their price ranges. The economic "
        "mechanism is based on the tendency for prices to move in the direction of the initial "
        "breakout due to inertia and follow-through sentiment."
    )

    def __init__(self, window: int = 15, consecutive_days: int = 3) -> None:
        self._window = window
        self._consecutive_days = consecutive_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_candidates: dict[str, float] = {}
        for symbol in view.symbols:
            close_series = [float(v) for v in history["adj_close"][symbol].drop_nulls().to_list()]
            if len(close_series) < self._window:
                continue
            bollinger_bands = _bollinger_bands(history, symbol)
            breakout_level = bollinger_bands[0]
            if close_series[-1] > breakout_level and all(
                close >= breakout_level for close in close_series[-self._consecutive_days:]
            ):
                momentum_score = sum(1 for close in close_series[-self._window:] if close > breakout_level)
                breakout_candidates[symbol] = momentum_score / len(close_series)

        picks: list[str] = sorted(breakout_candidates, key=breakout_candidates.get, reverse=True)[:20]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _bollinger_bands(history: pl.DataFrame, symbol: str) -> tuple[float]:
    close_series = history["adj_close"][symbol].drop_nulls().to_list()
    window = min(len(close_series), 20)
    if window < 5:
        return (None,)
    mean_close = sum(close_series[-window:]) / window
    std_dev = ((sum((close - mean_close) ** 2 for close in close_series[-window:]) / window) ** 0.5)
    upper_band = mean_close + 2 * std_dev
    lower_band = mean_close - 2 * std_dev
    return (upper_band,)