from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy aims to identify stocks with both high recent volatility and positive "
        "price momentum. High volatility suggests that the stock is experiencing heightened "
        "trading activity, potentially driven by news or events. Positive price momentum indicates "
        "that the stock has been performing well recently, suggesting continued buying interest."
    )

    def __init__(self, volatility_window: int = 10, momentum_window: int = 20) -> None:
        self._volatility_window = volatility_window
        self._momentum_window = momentum_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._volatility_window, self._momentum_window))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volatility_symbols = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue
            vol_close = (history.select(pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).sort("session_date", descending=True)
                         .head(self._volatility_window)["adj_close"])
            volatility = vol_close.std()
            momentum_close = (history.filter(
                pl.col("symbol") == symbol
            ).select(pl.col("adj_close")).sort("session_date", descending=True)
                              .head(self._momentum_window)["adj_close"])
            if volatility > 0.01 and momentum_close[-1] / momentum_close[0] >= 1.05:
                high_volatility_symbols.append(symbol)

        high_volatility_symbols = list(set(high_volatility_symbols))[:5]
        if not high_volatility_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_volatility_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in high_volatility_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest