from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves often indicate strong market sentiment. "
        "These moves are more likely to continue in the direction of their initial move if they "
        "are supported by high trading volumes. This strategy aims to capture such movements."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_close_series = history[symbol].select(pl.col("adj_close").to_list())
            volume_series = history[symbol].select(pl.col("volume").to_list())

            # Calculate daily returns and moving average of returns
            returns = [float(v1 / v2 - 1.0) for v1, v2 in zip(adj_close_series.to_list()[1:], adj_close_series.to_list()[:-1])]
            avg_return = sum(returns) / len(returns)

            # Filter high volume days where the return is above average
            high_volume_days = [i for i, vol in enumerate(volume_series.to_list()) if vol > 2 * history[symbol].select(pl.col("volume").mean().item())]
            filtered_returns = [ret for i, ret in enumerate(returns) if i in high_volume_days]

            # Check if the last return is above average and positive
            if len(filtered_returns) > 0 and filtered_returns[-1] > avg_return:
                high_volume_symbols.append(symbol)

        weights = {s: 1.0 / len(high_volume_symbols) for s in high_volume_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest