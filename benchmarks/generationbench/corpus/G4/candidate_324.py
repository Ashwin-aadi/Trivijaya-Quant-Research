from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "By focusing on stocks with strong recent performance (cross-sectional momentum), the strategy aims to "
        "capitalize on persistently high-performing stocks in the Indian market. This is based on the observation that past "
        "winners are likely to outperform again due to sustained positive fundamentals or investor sentiment."
    )

    def __init__(self, window: int = 52) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        stock_returns = []
        for symbol in view.symbols:
            row = history.filter(pl.col("symbol") == symbol).select(
                pl.col("session_date"), "close", "low"
            )
            if row.height < self._window:
                continue
            price_series = [float(v) for v in row["close"].drop_nulls().to_list()]
            low_price_series = [float(v) for v in row["low"].drop_nulls().to_list()]
            latest_close = view.latest_close()[symbol]
            lowest_price = min(low_price_series)
            if lowest_price == 0:
                continue
            return_52w = (latest_close - lowest_price) / lowest_price * 100.0
            stock_returns.append((symbol, return_52w))

        ranked_stocks = sorted(stock_returns, key=lambda x: x[1], reverse=True)
        top_stocks = ranked_stocks[: min(30, len(view.symbols))]

        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [x[0] for x in top_stocks]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest