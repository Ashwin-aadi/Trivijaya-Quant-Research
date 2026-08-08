from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "High volatility often precedes a trend reversal. By scaling trades with historical "
        "volatility, one can capture more returns during trending periods while reducing exposure "
        "to range-bound markets where no clear direction exists."
    )

    def __init__(self, window: int = 20, scaling_factor: float = 1.5) -> None:
        self._window = window
        self._scaling_factor = scaling_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_volatility: dict[str, float] = {}
        for symbol in view.symbols:
            prices = history.filter(pl.col("symbol") == symbol)[["adj_close"]]
            log_returns = (prices / prices.shift(1) - 1).drop_nulls()
            volatility = log_returns.std().item() * self._scaling_factor
            symbol_volatility[symbol] = volatility

        sorted_symbols = [
            s for _, s in sorted(symbol_volatility.items(), key=lambda item: -item[1])
        ]
        top_n_symbols = sorted_symbols[: int(self._window / 2)]

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