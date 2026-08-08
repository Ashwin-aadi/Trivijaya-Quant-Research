from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects the top-performing stocks relative to the NIFTY 100 index "
        "based on their returns over a lookback period. The idea is that outperforming "
        "stocks may continue to outperform due to market inefficiencies or sector-specific "
        "factors."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(view.symbols) < 2:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window).sort("session_date", descending=True)

        index_returns = (
            closes.select(pl.col(view.as_of.strftime("%Y-%m-%d")) / pl.col("adj_close") - 1)
            .to_series()
            .mean()
        )
        stock_returns = [
            (float(closes[symbol].to_list()[0]) / float(view.latest_close()[symbol]) - 1)
            for symbol in view.symbols
        ]

        top_stocks = sorted(
            zip(view.symbols, stock_returns), key=lambda x: x[1], reverse=True
        )[:5]

        weight = 1.0 / len(top_stocks)
        return Signal(information_available_at=stamp, weights={s: weight for s in [t[0] for t in top_stocks]})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest