from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are typically more responsive to market conditions and can "
        "provide a smoother performance. By equal-weighting these liquid stocks, we aim to "
        "capitalize on their consistent trading behavior without overconcentrating risk in "
        "less active securities."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols with insufficient trading volume
        min_volume_threshold = 10_000_000
        filtered_history = (
            history.filter(pl.col("volume").gt(min_volume_threshold))
            .group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("avg_price"),
                pl.count().alias("trade_count"),
            )
        )

        # Ensure we have enough symbols to proceed
        if filtered_history.height < 5:
            return Signal(information_available_at=stamp, weights={})

        # Equal weight allocation among the top liquid stocks
        n_symbols = min(filtered_history.height, 10)
        equal_weight = 1.0 / n_symbols
        liquidity_picks: list[str] = filtered_history.select("symbol").to_dict()[0]

        return Signal(
            information_available_at=stamp,
            weights={s: equal_weight for s in liquidity_picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest