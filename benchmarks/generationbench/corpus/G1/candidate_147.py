from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting involves assigning weights to stocks based on their "
        "trading volume. Higher liquidity is associated with more confidence in the stock's price "
        "and easier execution of trades. This approach aims to balance risk by distributing capital "
        "evenly among highly liquid stocks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_weights = (
            history.group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
                pl.col("adj_close").mean().alias("avg_price"),
            )
            .with_columns(
                (pl.col("total_volume") / pl.sum("total_volume")).alias("weight")
            )
            .select(["symbol", "weight"])
        )

        weights_dict = (
            liquidity_weights.to_pandas()
            .set_index("symbol")["weight"]
            .to_dict()
        )

        return Signal(
            information_available_at=stamp, weights=weights_dict
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest