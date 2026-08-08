from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices and returns eventually return to the long-term "
        "average. In a short-horizon context, recent extreme price movements are likely to reverse."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("adj_close").mean().alias("mean")),
        ).to_vertical()

        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores: list[float] = []
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in closes.columns:
                continue
            close_values = [float(v) for v in closes[symbol].to_list()]
            mean_close_value = float(mean_close.filter(pl.col("symbol") == symbol)[0]["mean"])
            score = (close_values[-1] - mean_close_value) / mean_close_value

            if abs(score) > 2:  # Considering scores with absolute value greater than 2 as extreme
                mean_reversion_scores.append(score)

        symbols_with_extreme_moves = [s for s in view.symbols if any(v != 0 for v in mean_reversion_scores)]
        weight = 1.0 / len(symbols_with_extreme_moves)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols_with_extreme_moves}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest