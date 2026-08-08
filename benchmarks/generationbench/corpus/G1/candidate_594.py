from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength compared to the NIFTY 100 index are likely "
        "to outperform in the near term. This strategy aims to identify such stocks by "
        "calculating their performance against the market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty100_close = view.closes().select(pl.col("^NSEI").alias("nifty100")).to_series()
        stock_closes = view.closes().select([pl.col(c) for c in view.symbols]).transpose().to_series()

        if stock_closes.is_null().sum() > 0 or nifty100_close.is_null():
            return Signal(information_available_at=stamp, weights={})

        nifty100_returns = (nifty100_close / nifty100_close.shift(1) - 1.0).to_list()
        stock_returns = [(stock_closes[i] / stock_closes[i-1] - 1.0) for i in range(1, len(stock_closes))]

        strength_ratio = [r_n / r_s for r_s, r_n in zip(stock_returns, nifty100_returns)]

        top_stocks = sorted(zip(view.symbols, strength_ratio), key=lambda x: -x[1])[:5]
        
        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest