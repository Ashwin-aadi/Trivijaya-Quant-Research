from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures that only the most liquid stocks are considered for "
        "equal-weighted portfolios. This reduces transaction costs and can improve risk-adjusted "
        "returns."
    )

    def __init__(self, min_trading_volume: float = 100_000) -> None:
        self._min_trading_volume = min_trading_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)  # Consider one year of data
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquid_symbols = [
            symbol for symbol in view.symbols if float(history[symbol]["volume"].mean()) >= self._min_trading_volume
        ]
        if not liquid_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquid_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquid_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest