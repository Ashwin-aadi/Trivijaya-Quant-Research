from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts that continue to be supported or resisted by price action can indicate "
        "long-term structural changes in the market. Identifying such continuations can lead "
        "to profitable trading opportunities."
    )

    def __init__(self, window: int = 20, support_lookback: int = 10) -> None:
        self._window = window
        self._support_lookback = support_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._support_lookback)

        if history.height < self._window + 1 or view.closes().height < 2:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            prices = [float(v) for v in history[history["symbol"] == symbol]["adj_close"].to_list()]
            support_prices = [
                float(v)
                for v in history[
                    (history["session_date"] >= stamp - pl.Duration(days=self._support_lookback))
                    & (history["symbol"] == symbol)
                ]["adj_close"].to_list()
            ]
            
            if len(prices) < self._window or len(support_prices) < 1:
                continue

            # Check for breakout
            last_price = prices[-1]
            if last_price >= max(prices):
                # Check support level after the breakout
                min_support = min(support_prices)
                for price in reversed(support_prices):
                    if price < min_support and last_price > price:
                        break
                    elif price == min_support and last_price <= price:
                        break
                else:
                    breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest