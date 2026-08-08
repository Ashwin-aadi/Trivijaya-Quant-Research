from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "A stock with a higher relative strength compared to the broader market "
        "indicates it is outperforming and may offer better risk-adjusted returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or f"{symbol}_return" not in history.columns:
                continue
            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_values) < self._window:
                continue

            returns = [(close_values[i] - close_values[i-1]) / close_values[i-1] if close_values[i-1] != 0 else 0.0 for i in range(1, self._window)]
            window_return = sum(returns)
            broad_market_return = history[f"{symbol}_return"].mean().item()
            
            strength = (window_return - broad_market_return) / broad_market_return
            symbol_strengths[symbol] = strength

        sorted_stocks = sorted(symbol_strengths.items(), key=lambda x: x[1], reverse=True)
        top_n_stocks = [stock for stock, _ in sorted_stocks[:5]]

        if not top_n_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_stocks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_n_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest