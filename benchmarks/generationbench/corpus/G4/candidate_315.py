from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the liquidity effect by focusing on highly liquid stocks "
        "and applying an equal weighting approach. Liquid stocks often have lower bid-ask spreads "
        "and smaller price movements, leading to reduced trading costs and potentially higher returns."
    )

    def __init__(self, window: int = 30, top_n: int = 100) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols that are not present in the entire lookback period
        symbols = [symbol for symbol in view.symbols if symbol in history.columns]

        if len(symbols) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        # Calculate average daily volume over the last 30 days for each stock
        avg_volume = (
            history[symbols]
            .select(
                pl.col("symbol"),
                (pl.col("volume").sum() / self._window).alias("avg_volume")
            )
            .sort("avg_volume", descending=True)
            .head(self._top_n)
        )

        # Select the top N most liquid stocks
        picks = avg_volume["symbol"].to_list()

        # Equal weighting for each selected stock
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