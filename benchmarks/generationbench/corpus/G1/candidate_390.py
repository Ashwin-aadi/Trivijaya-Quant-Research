from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects the top-performing stocks relative to the NIFTY 100 index. "
        "Historically, such stocks have shown resilience and outperformance during market cycles."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_closes = view.history().select("NIFTY_CLOSE").drop_nulls()
        nifty_returns = (nifty_closes / nifty_closes.shift(1) - 1.0).alias("NIFTY_RTN")

        symbols = [col for col in closes.columns if "symbol_" not in col]
        stock_returns = (closes[symbols] / closes[symbols].shift(1) - 1.0).transpose()

        stock_nifty_ratio = stock_returns.div(nifty_returns, fill_value=0)
        ranked_ratios = stock_nifty_ratio.transpose().select(
            pl.col("session_date").alias("date"),
            (pl.col(symbols) / pl.col("NIFTY_RTN").rank(method="dense", descending=True)).alias("strength"),
        )

        top_symbols = [row[1].item() for row in ranked_ratios.sort("strength", descending=True).select("symbols").rows()]
        top_symbols = top_symbols[: self._top_n]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest