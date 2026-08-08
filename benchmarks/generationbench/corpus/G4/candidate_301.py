from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion50d(Strategy):
    rationale = (
        "This strategy aims to capitalize on short-horizon mean reversion in the Indian market. "
        "Stock prices tend to revert to their historical average due to market inefficiencies and "
        "behavioral biases. By identifying stocks that have deviated significantly from their 50-day moving averages, we can exploit this persistence."
    )

    def __init__(self, window: int = 50, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        moving_average = (
            view.history()
            .group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("ma"),
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().mean().alias("volatility"),
            )
        )

        price_to_moving_average_ratio = (
            view.closes()
            .join(moving_average, on="symbol")
            .with_columns(
                (pl.col("close") / pl.col("ma")).alias("p_m_a_ratio"),
                ((pl.col("adj_close").shift(1) - pl.col("ma")) / pl.col("volatility")).abs().rank(method="dense", descending=True).alias("reversion_rank")
            )
        )

        ranked = price_to_moving_average_ratio.sort("reversion_rank").select(["symbol", "p_m_a_ratio"])

        top_symbols = [row["symbol"] for row in ranked.to_dict(orient="records")[: self._top_n]]

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