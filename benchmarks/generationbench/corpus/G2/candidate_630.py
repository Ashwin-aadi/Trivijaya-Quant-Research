from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength tend to outperform over the medium term. "
        "Relative strength is defined here as the ratio of a stock's price momentum to the "
        "momentum of the market index. Stocks that consistently have stronger relative "
        "strength are more likely to continue performing well."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        market_index_symbol = "NIFTY100"  # Assuming NIFTY100 is the index symbol
        if market_index_symbol not in view.history().column_names:
            return Signal(information_available_at=stamp, weights={})

        market_history = view.history(symbol=market_index_symbol)
        market_moments = (
            market_history.sort("session_date")
            .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("r"))
            .select(pl.col("r").to_list())
        )
        if len(market_moments.to_list()) < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_momentum = max(market_moments.to_list())

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.column_names:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window * 2:
                continue

            stock_moments = (pl.DataFrame(values).sort("session_date")
                             .with_columns((pl.col(0) / pl.col(0).shift(self._window) - 1.0).alias("r"))
                             .select(pl.col("r").to_list()))
            if len(stock_moments.to_list()) < self._window:
                continue

            stock_momentum = max(stock_moments.to_list())
            relative_strength = (stock_momentum / market_momentum) * 100
            signals[symbol] = relative_strength

        top_symbols = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in [symbol for symbol, _ in top_symbols]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest