from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the relationship between market trends and volatility. "
        "High volatility periods often precede trend continuation or reversal, while low "
        "volatility can indicate established trends. By scaling trades based on historical "
        "volatility, the strategy aims to capture profits from trending behavior while "
        "mitigating risk."
    )

    def __init__(self, window: int = 20, top_n: int = 15) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility_scaled_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "adj_close" not in history.columns:
                continue

            adj_closes = [float(v) for v in history[symbol][1:self._window + 1]["adj_close"].to_list()]
            returns = [(adj_closes[i] - adj_closes[i-1]) / adj_closes[i-1] if adj_closes[i-1] != 0 else 0.0 for i in range(1, self._window)]
            volatility = pl.DataFrame({"returns": returns}).select((pl.col("returns").std()).alias("volatility")).item()

            if volatility > 0:
                scaled_signal = (history[history["symbol"] == symbol]["adj_close"].max() - adj_closes[-1]) / volatility
            else:
                scaled_signal = 0.0

            volatility_scaled_signals[symbol] = scaled_signal

        ranked_symbols = sorted(volatility_scaled_signals.items(), key=lambda x: x[1], reverse=True)
        top_n_symbols = [symbol for symbol, _ in ranked_symbols[:self._top_n]]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest