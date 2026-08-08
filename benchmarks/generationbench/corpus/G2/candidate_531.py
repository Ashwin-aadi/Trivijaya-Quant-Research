from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are typically more responsive to market conditions and have "
        "lower transaction costs. By equally weighting high-liquidity stocks, the strategy aims "
        "to capture the benefits of these stocks without overconcentration in any single stock."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter for symbols with sufficient trading volume
        min_volume = 10_000_000  # Adjust based on market norms
        high_liquidity_symbols = (
            history.select(
                pl.col("symbol"), pl.sum(pl.col("volume")).alias("total_volume")
            )
            .filter(pl.col("total_volume") > min_volume)
            .select(pl.col("symbol"))
            .to_series()
            .to_list()
        )

        if not high_liquidity_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weight allocation among selected symbols
        num_selected = len(high_liquidity_symbols)
        weight_per_symbol = 1.0 / num_selected

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight_per_symbol for symbol in high_liquidity_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest