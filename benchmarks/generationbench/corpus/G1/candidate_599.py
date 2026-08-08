from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeAndMomentum(Strategy):
    rationale = (
        "This strategy combines volume and momentum indicators to identify "
        "overbought or oversold conditions. High volume with high close relative to the 20-day mean suggests "
        "strong buying pressure, while low volume with a high close indicates potential profit-taking."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history["adj_close"].to_list()]
        volumes = [float(v) for v in history["volume"].to_list()]

        mean_close = sum(closes[-self._window:]) / self._window
        breakout_threshold = 1.05 * mean_close

        potential_signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].drop_nulls().to_list()) < self._window + 1:
                continue

            recent_closes = [float(v) for v in history[symbol]["adj_close"].drop_nulls().to_list()[-self._window:]]
            recent_volumes = [float(v) for v in history[symbol]["volume"].drop_nulls().to_list()[-self._window:]]

            if (
                max(recent_closes) >= breakout_threshold
                and recent_volumes[0] > sum(recent_volumes) / self._window
            ):
                potential_signals.append(symbol)

        if not potential_signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(potential_signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in potential_signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest