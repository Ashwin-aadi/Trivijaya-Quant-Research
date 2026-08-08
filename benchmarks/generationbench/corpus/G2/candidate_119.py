from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term and long-term momentum signals can filter out noise and "
        "potentially improve returns. Short-term momentum captures recent price action, "
        "while long-term momentum reflects sustained trends over a longer period."
    )

    def __init__(self, short_window: int = 20, long_window: int = 60) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._long_window + self._short_window - 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        recent_closes = history.select(
            pl.col("symbol").alias("symbol"),
            pl.col("adj_close").tail(self._short_window).mean().alias("recent_mean"),
            pl.col("adj_close").tail(self._long_window).mean().alias("long_term_mean"),
        )

        signals: list[str] = []
        for symbol in view.symbols:
            recent_mean = float(recent_closes[recent_closes["symbol"] == symbol]["recent_mean"])
            long_term_mean = float(
                recent_closes[recent_closes["symbol"] == symbol]["long_term_mean"]
            )
            if (
                (recent_mean > 1.02 * long_term_mean)
                or
                (recent_mean < 0.98 * long_term_mean)
            ):
                signals.append(symbol)

        signal_dict = {s: 0.5 for s in signals}
        if not signal_dict:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signal_dict.values())
        adjusted_weights = {k: v / total_weight for k, v in signal_dict.items()}
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight
                for symbol, weight in adjusted_weights.items()
                if weight > 0.01
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest