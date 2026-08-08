from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBreakoutAndVolatility(Strategy):
    rationale = (
        "This strategy combines volume breakout with historical volatility to identify "
        "high-activity stocks that are likely experiencing a significant price movement."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            volume_history = [float(v) for v in history[f"{symbol}.volume"].to_list()]
            adj_close_history = [float(v) for v in history[f"{symbol}.adj_close"].to_list()]

            if len(volume_history) < self._window or len(adj_close_history) < self._window:
                continue

            volume_mean = pl.DataFrame({"v": volume_history}).select(
                (pl.col("v").mean()).alias("volume_mean")
            ).height
            adj_close_mean = pl.DataFrame({"c": adj_close_history}).select(
                (pl.col("c").mean()).alias("adj_close_mean")
            ).height

            if volume_mean == 0 or adj_close_mean == 0:
                continue

            daily_returns = [
                ((adj_close_history[i + 1] - adj_close_history[i]) / adj_close_history[i])
                for i in range(len(adj_close_history) - 1)
            ]
            volatility = (sum([abs(ret) for ret in daily_returns]) / len(daily_returns)) * 100

            if volume_history[-1] > volume_mean and volatility > 3:
                high_volume_symbols.append(symbol)

        high_volume_symbols = high_volume_symbols[: self._top_n]
        if not high_volume_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(high_volume_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in high_volume_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest