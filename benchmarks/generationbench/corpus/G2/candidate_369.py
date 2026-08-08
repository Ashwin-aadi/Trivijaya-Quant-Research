from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Lower volatility assets tend to have higher expected returns due to risk premium. "
        "By tilting our portfolio towards lower-volatility stocks, we can potentially capture "
        "this risk premium."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            returns = (
                history[f"{symbol}_close"]
                .to_list()[1:]
                .pct_change()
                .drop_nulls()
                .to_list()
            )
            if len(returns) < self._window - 1:
                continue
            volatility = (sum([r**2 for r in returns]) / (self._window - 1)) ** 0.5
            volatilities[symbol] = float(volatility)

        sorted_symbols = [k for k, v in sorted(volatilities.items(), key=lambda item: item[1])]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest