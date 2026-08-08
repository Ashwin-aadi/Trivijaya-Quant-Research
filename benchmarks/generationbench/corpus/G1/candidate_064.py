from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "This strategy aims to identify stocks that have deviated significantly from their "
        "trailing average price and are likely to revert. It focuses on mean reversion "
        "principles within the Indian market."
    )

    def __init__(self, window: int = 50, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        if closes.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"]]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.filter(pl.col("symbol").is_in(symbols))
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns((pl.col("adj_close") / pl.col("mean")).alias("deviation"))
        )

        picks: list[str] = []
        for symbol in symbols:
            mean_val = float(mean_close.filter(pl.col("symbol") == symbol)["mean"][0])
            if symbol not in closes.columns or len(closes[symbol].to_list()) < self._window + 1:
                continue
            last_adj_close = float(closes[symbol][-1])
            deviation = (last_adj_close / mean_val) - 1.0

            if abs(deviation) >= self._threshold:
                picks.append(symbol)

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest