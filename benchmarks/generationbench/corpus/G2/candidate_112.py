from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Trailing reversion seeks to identify stocks that have deviated significantly from their "
        "recent price levels. When the current price returns to a recent mean, it suggests a reversal in trend, offering an opportunity for profit."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_close = history.select(
            pl.col("adj_close").mean().alias("avg_close")
        ).into_dataframe()
        avg_close = avg_close.with_columns(pl.lit(stamp).alias("session_date"))

        merged = history.join(avg_close, on="session_date", how="inner")
        merged = merged.with_columns(
            (pl.col("adj_close") - pl.col("avg_close")).abs().alias("deviation")
        )

        top_deviants = (
            merged.sort("deviation", descending=True)
            .select(["symbol", "deviation"])
            .head(self._window)
        )
        symbols = [row["symbol"] for row in top_deviants.to_dicts()]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest