from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-screened equal weighting involves selecting the most liquid stocks and "
        "allocating equal weights to them. Liquid stocks are less prone to large price "
        "fluctuations due to trading volume."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily volume for each symbol
        daily_volumes = (
            history.group_by("symbol")
                   .agg(pl.col("volume").sum().alias("total_volume"))
        )

        # Get the top N symbols by total volume
        liquidity_sorted_symbols = daily_volumes.sort("total_volume", descending=True)
        top_n_symbols = [str(symbol) for symbol in liquidity_sorted_symbols.head(self._window)["symbol"].to_list()]

        # Equal weighting among the selected symbols
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest