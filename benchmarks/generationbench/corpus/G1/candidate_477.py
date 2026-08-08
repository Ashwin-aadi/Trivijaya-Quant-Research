from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often considered less risky and may offer better risk-adjusted "
        "returns over the long term. By tilting the portfolio towards lower volatility stocks, we aim to "
        "reduce overall portfolio risk while potentially enhancing returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            returns = (history[symbol].drop_nulls().to_list()[1:] / 
                       history[symbol].shift(1).drop_nulls().to_list() - 1.0)
            volatilities[symbol] = ((pl.Series(returns) ** 2).mean()) ** 0.5

        if not volatilities:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = [k for k, v in sorted(volatilities.items(), key=lambda item: item[1])]
        top_n_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest