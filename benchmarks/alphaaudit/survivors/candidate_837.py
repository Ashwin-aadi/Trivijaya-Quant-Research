from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often underpriced relative to their high-volatility peers. "
        "By tilting our portfolio towards low-volatility equities, we aim to capture excess returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        )

        # Group by symbol and calculate standard deviation of returns over the window
        volatility_df = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("r").std().alias("volatility"))
            )
            .sort("volatility", descending=False)
            .select(["symbol", "volatility"])
        )

        # Select top N low-volatility stocks
        num_symbols = min(len(view.symbols), 5)
        low_vol_symbols = volatility_df.head(num_symbols)["symbol"].to_list()

        if not low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(low_vol_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in low_vol_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest