from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High-liquidity stocks are likely to have more efficient pricing. By investing "
        "equally in the most liquid stocks, we aim to benefit from this efficiency and avoid "
        "the bid-ask spread costs that can be higher for less liquid stocks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the volume-weighted average price for each symbol
        volume_wap_df = (
            history.with_columns(
                (pl.col("adj_close") * pl.col("volume")).alias("vwap_adj_close"),
                (pl.col("volume").sum()).alias("total_volume"),
            )
            .group_by("symbol")
            .agg((pl.col("vwap_adj_close").sum() / pl.col("total_volume")).alias("wavg"))
            .sort("wavg", descending=True)
        )

        # Get the top symbols by liquidity
        top_symbols = volume_wap_df.select(["symbol"]).head(10).to_dict(as_series=False)

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(top_symbols)
        weights = {s: weight_per_symbol for s in top_symbols["symbol"]}

        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest