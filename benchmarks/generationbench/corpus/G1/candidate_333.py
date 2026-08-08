from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks have historically outperformed high-volatility stocks over the long term. "
        "By tilting our portfolio towards low volatility, we aim to capture this risk premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the rolling standard deviation over the last 20 days
        volatilities: pl.DataFrame = (
            history.groupby("symbol")
                   .agg((pl.col("adj_close").std().alias("volatility")))
                   .sort("volatility", descending=False)
        )

        if volatilities.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Select the lowest volatility stocks
        picks: list[str] = [str(s) for s in volatilities["symbol"].to_list()[:5]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest