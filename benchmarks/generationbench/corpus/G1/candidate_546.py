from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects the stocks that have performed better than the overall market "
        "in a given period. The idea is to overweight these stocks in the portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_closes = view.closes(lookback=self._window)
        symbols = [col for col in view.symbols if col in market_closes.columns]

        market_returns = (market_closes[symbols] / market_closes[symbols].shift(1) - 1.0).to_series()
        stock_returns = (view.closes(symbols=symbols, lookback=self._window) / history.shift(self._window + 1)["adj_close"] - 1.0).to_series()

        strength_ratio = stock_returns.to_frame().join(market_returns.to_frame(), on="session_date")
        strength_ratio = strength_ratio.with_columns(
            (strength_ratio[f"{symbols[0]}_return"] / strength_ratio["market_return"]).alias("strength_ratio"),
        )

        top_stocks = strength_ratio.sort("strength_ratio", descending=True).select("symbol").to_series().head(5).to_list()
        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest