from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility periods often precede trend reversals. By focusing on the relative "
        "volatility of stocks within a market index, we can identify potential leaders that may "
        "reverse trends, allowing us to profit from subsequent mean reversion or continuation."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "session_date" not in history.columns:
                continue
            open_prices = [float(v) for v in history[symbol + "_open"].drop_nulls().to_list()]
            close_prices = [float(v) for v in history[symbol + "_close"].drop_nulls().to_list()]
            if len(open_prices) < self._window or len(close_prices) < self._window:
                continue

            returns = [(close / open - 1.0) for close, open in zip(close_prices, open_prices)]
            volatility_factor = sum(abs(r) for r in returns) / len(returns)
            volatility_factors[symbol] = volatility_factor

        sorted_symbols = [symbol for symbol, _ in sorted(volatility_factors.items(), key=lambda item: -item[1])]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = sorted_symbols[0]
        weight = 1.0
        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest