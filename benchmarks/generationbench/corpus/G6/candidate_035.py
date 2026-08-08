from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class IntegratedValueMomentum(Strategy):
    rationale = (
        "This strategy combines Value (low Price-to-Book ratio) and Momentum (positive price trend over a recent period), "
        "providing a balanced approach to stock selection. By integrating both factors, we aim to identify stocks with strong "
        "fundamental and technical signals."
    )

    def __init__(self, value_window: int = 20, momentum_window: int = 30) -> None:
        self._value_window = value_window
        self._momentum_window = momentum_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._value_window + self._momentum_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        value_scores: dict[str, float] = {}
        momentum_scores: dict[str, float] = {}

        for symbol in view.symbols:
            daily_data = history.filter(pl.col("symbol") == symbol)
            closes = [float(v) for v in daily_data["adj_close"].to_list()]
            if len(closes) < self._value_window + self._momentum_window:
                continue

            # Value Score: Inverse of Price-to-Book ratio
            book_values = [float(v) for v in history.filter(pl.col("symbol") == symbol)["book_value"].to_list()]
            value_ratio = sum(closes[-self._value_window:]) / sum(book_values[-self._value_window:])
            value_scores[symbol] = 1.0 / value_ratio

            # Momentum Score: Price change over momentum_window
            momentum_change = (closes[-1] - closes[0]) / closes[0]
            if momentum_change > 0:
                momentum_scores[symbol] = abs(momentum_change)

        combined_scores = {symbol: value_scores.get(symbol, 0) + momentum_scores.get(symbol, 0) for symbol in view.symbols}
        ranked_symbols = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_20_percent = int(len(ranked_symbols) * 0.2)

        if top_20_percent == 0:
            return Signal(information_available_at=stamp, weights={})

        picks = [symbol for symbol, score in ranked_symbols[:top_20_percent]]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest