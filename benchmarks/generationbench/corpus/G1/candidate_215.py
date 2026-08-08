from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines the recent strength of a stock with its volatility to "
        "identify potentially profitable opportunities. Strong stocks with higher volatility "
        "may offer more immediate gains."
    )

    def __init__(self, window: int = 20, threshold_volatility: float = 0.1) -> None:
        self._window = window
        self._threshold_volatility = threshold_volatility

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            (closes["adj_close"] / closes["adj_close"].shift(1) - 1.0)
            .drop_nulls()
            .to_list()
        )
        if len(returns) < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate volatility
        volatility = (
            pl.DataFrame({"return": returns})
            .with_columns((pl.col("return").abs().mean()).alias("volatility"))
            .select("volatility")
            .to_series()
            .to_list()[0]
        )

        if volatility < self._threshold_volatility:
            return Signal(information_available_at=stamp, weights={})

        # Identify strong symbols
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            if max(values) - min(values) > (max(values) + min(values)) * 0.15:
                picks.append(symbol)

        # Select top symbols based on the combination of strength and volatility
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest