from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining the 20-day momentum and the 50-day volatility can provide a more robust signal. "
        "Momentum helps identify trending stocks, while low volatility suggests relative stability."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 50) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window + self._volatility_window)
        if closes.height < self._momentum_window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_values) < self._momentum_window + self._volatility_window:
                continue

            # Calculate 20-day momentum
            recent_close = float(close_values[-1])
            first_close = float(close_values[0])
            momentum_score = (recent_close - first_close) / first_close
            momentum_scores[symbol] = momentum_score

            # Calculate 50-day volatility
            vol_value = pl.DataFrame({
                "log_ret": [pl.col("adj_close").log() - pl.col("adj_close").shift(1).log()
                            for adj_close in close_values]
            })["log_ret"].sum().item()
            volatility_scores[symbol] = vol_value

        # Filter symbols based on momentum and volatility scores
        selected_symbols = [
            symbol for symbol, score in momentum_scores.items()
            if score > 0.1 and volatility_scores[symbol] < -0.05
        ]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest