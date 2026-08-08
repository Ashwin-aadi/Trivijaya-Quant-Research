from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class JanuaryEffect(Strategy):
    rationale = (
        "This strategy exploits the January Effect in the Indian market by identifying "
        "stocks that historically perform well during early January. It involves buying a "
        "diversified portfolio of top-ranked stocks from December 20th to January 20th, "
        "holding them until February, and then selling for potential upside."
    )

    def __init__(self, window: int = 5 * 12 + 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Filter to the relevant dates
        start_date = date(stamp.year - 1, 12, 20)
        end_date = date(stamp.year + 1, 2, 20)

        filtered_history = history.filter(
            (pl.col("session_date") >= start_date) & (pl.col("session_date") <= end_date)
        )

        if filtered_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate cumulative returns for each stock
        filtered_history = (
            filtered_history.with_columns(
                ((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("daily_return"))
            )
            .group_by("symbol")
            .agg(pl.col("daily_return").sum().alias("cumulative_return"))
        )

        # Rank stocks based on cumulative return
        ranked = filtered_history.sort("cumulative_return", descending=True)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in ranked["symbol"]:
                continue
            picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

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