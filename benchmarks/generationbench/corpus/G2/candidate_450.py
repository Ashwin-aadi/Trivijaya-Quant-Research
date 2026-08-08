from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed their peers in the past are likely to continue "
        "outperforming due to a combination of momentum and investor preference. This strategy "
        "aims to identify such stocks by comparing their recent returns against the broader market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        market_returns = (closes[closes.columns[-1]] / closes[closes.columns[0]].shift(1) - 1.0).alias("market_return")
        stock_returns = {symbol: (closes[symbol] / closes[closes.columns[0]].shift(1) - 1.0).alias(f"return_{symbol}")
                         for symbol in view.symbols}
        combined_df = pl.concat([pl.DataFrame(market_returns), *stock_returns.values()], how="horizontal")

        mean_market_return = combined_df["market_return"].mean()
        stock_strengths = [float(combined_df[combined_df[f"return_{symbol}"] > mean_market_return][f"return_{symbol}"].sum())
                           for symbol in view.symbols]

        strong_stocks = sorted(zip(view.symbols, stock_strengths), key=lambda x: x[1], reverse=True)[:5]
        if not strong_stocks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(strong_stocks)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in strong_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest