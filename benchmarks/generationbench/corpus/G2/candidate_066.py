from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "The relative strength (RS) strategy selects stocks that have outperformed the market "
        "over a recent period. Stocks with higher relative strength are expected to continue "
        "outperforming due to strong fundamentals or investor sentiment."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate relative strength
        symbol_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in history["symbol"]:
                continue

            close_values = [float(v) for v in closes[symbol].to_list()]
            if len(close_values) < self._window:
                continue

            market_close_values = [
                float(close) for _, close in history.filter(pl.col("symbol") == symbol).rows()
            ]
            if len(market_close_values) < self._window:
                continue

            # Calculate returns
            stock_returns = [close / prev_close - 1.0 for close, prev_close in zip(close_values[1:], close_values[:-1])]
            market_returns = [
                close / prev_close - 1.0 for close, prev_close in zip(market_close_values[1:], market_close_values[:-1])
            ]

            # Calculate relative strength
            rel_strength = sum(stock_return > market_return for stock_return, market_return in zip(stock_returns, market_returns))
            symbol_strengths[symbol] = rel_strength / self._window

        # Sort symbols by their relative strength and select top N
        sorted_symbols = [
            (symbol, strength) for symbol, strength in symbol_strengths.items() if strength >= self._threshold
        ]
        sorted_symbols.sort(key=lambda x: x[1], reverse=True)

        picks = [symbol for symbol, _ in sorted_symbols[:5]]  # Take top 5

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