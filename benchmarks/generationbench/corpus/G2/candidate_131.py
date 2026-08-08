from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines volume and momentum indicators to identify undervalued stocks. "
        "High trading volume often signals a strong interest in the stock, while recent price "
        "gains indicate that the market is positive about the company's future prospects."
    )

    def __init__(self, momentum_window: int = 10, volume_threshold: float = 100_000) -> None:
        self._momentum_window = momentum_window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window)
        if history.height < self._momentum_window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            # Calculate momentum
            adj_close_series = pl.col("adj_close")
            daily_returns = (adj_close_series / adj_close_series.shift(1) - 1.0).alias("r")
            momentum = daily_returns.sum().round(4)

            # Check volume threshold
            volume_df = view.closes(lookback=self._momentum_window)
            if symbol not in volume_df.columns:
                continue
            volume = float(volume_df[symbol].sum())

            if volume >= self._volume_threshold and momentum > 0.05:
                picks.append(symbol)

        picks = picks[:10]
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