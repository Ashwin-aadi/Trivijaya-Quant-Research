from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength tend to outperform over the short term. "
        "This is based on the idea that stocks with strong performance are likely to continue "
        "outperforming as market sentiment remains positive for these companies."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(view.symbols) <= 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the average close over the lookback period
        avg_closes = (
            closes.groupby("symbol")
            .agg(
                pl.col("adj_close").mean().alias("avg_adj_close"),
            )
            .collect()
        )

        # Compute relative strength by dividing each stock's current close by its average close
        rel_strength = (
            view.closes(lookback=None)
            .join(avg_closes, on="symbol")
            .with_columns(
                (pl.col("adj_close") / pl.col("avg_adj_close")).alias("rel_strength"),
            )
            .sort("rel_strength", descending=True)
            .select(["session_date", "symbol", "rel_strength"])
            .to_pandas()
        )

        top_symbols = rel_strength["symbol"].head(5).tolist()

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={(s): weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest