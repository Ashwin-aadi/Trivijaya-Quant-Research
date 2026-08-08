from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks tend to offer more stable returns over the long term. "
        "By tilting our portfolio towards low volatility assets, we aim to reduce overall risk "
        "while still capturing market returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].unique()) < 2:
                continue
            daily_returns = (
                (history["adj_close"].shift(-1) / history["adj_close"]) - 1.0
            ).to_list()
            volatilities[symbol] = pl.Series(daily_returns).std().item()

        sorted_symbols = [
            s for s, v in sorted(volatilities.items(), key=lambda item: item[1])
        ]
        top_n_symbols = sorted_symbols[:5]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest