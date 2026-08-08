from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumStrategy(Strategy):
    rationale = (
        "To exploit cross-sectional momentum in the Indian equity market, this strategy "
        "identifies top-performing stocks over the past 3 to 6 months and allocates capital "
        "proportionally towards these 'winners'. The economic mechanism is based on the "
        "tendency of past winners to outperform due to slow portfolio adjustments and investor biases."
    )

    def __init__(self, window: int = 120, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate cumulative returns for each stock
        rank_column = "cumulative_return"
        ranking = (
            history.select(pl.col("symbol"))
            .join(closes, on="symbol", how="inner")
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias(rank_column)
            )
            .sort(rank_column, descending=True)
        )

        # Select top N stocks based on cumulative returns
        picks: list[str] = ranking.select("symbol").head(self._top_n).to_list()[0]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        # Allocate capital proportionally to the selected stocks
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest