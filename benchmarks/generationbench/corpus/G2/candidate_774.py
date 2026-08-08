from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets that have outperformed the broader market over a defined period are "
        "more likely to continue their strong performance due to momentum effects."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_closes = history.select(
            pl.col("symbol").filter(pl.col("symbol").is_in(view.symbols)),
            "adj_close",
        )
        average_nifty_close = (
            nifty_closes.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("avg"))
            .with_column(pl.col("avg") / pl.col("avg").max() * 100.0)
            .sort("avg", descending=True)
        )

        if average_nifty_close.height < view.symbols.length:
            return Signal(information_available_at=stamp, weights={})

        top_n_symbols = [row["symbol"] for row in average_nifty_close.head(self._window)]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(top_n_symbols, [weight] * len(top_n_symbols))),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest