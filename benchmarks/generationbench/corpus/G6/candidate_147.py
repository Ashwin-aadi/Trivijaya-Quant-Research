from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy selects stocks based on high liquidity to ensure robust trading and "
        "diversifies the portfolio by equal weighting, rebalancing daily. It aims to balance "
        "liquidity screening with a simple risk management rule."
    )

    def __init__(self, min_daily_trading_value: int = 10_000_000, min_volume: int = 500_000, max_positions: int = 30) -> None:
        self._min_daily_trading_value = min_daily_trading_value
        self._min_volume = min_volume
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=90)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_trading_value = (history[symbol]["close"] * history[symbol]["volume"]).sum()
            volume = history[symbol]["volume"].mean()
            if daily_trading_value >= self._min_daily_trading_value and volume > self._min_volume:
                high_volume_symbols.add(symbol)

        if len(high_volume_symbols) < 1:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = list(high_volume_symbols)
        selected_symbols = sorted_symbols[:self._max_positions]
        weight = 1.0 / self._max_positions
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