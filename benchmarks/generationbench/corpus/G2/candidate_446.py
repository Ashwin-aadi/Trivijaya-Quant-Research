from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "High-volatility assets often experience mean reversion in the short term. "
        "However, during trending periods, high volatility can indicate a continuation of the trend. "
        "By scaling the trend following strategy with historical volatility, we aim to capture trends more effectively."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * 2:
            return Signal(information_available_at=stamp, weights={})

        # Compute daily returns
        daily_returns = (closes / closes.shift(1) - 1.0).rename("r")

        # Calculate historical volatility
        volatilities = (
            daily_returns.groupby(pl.col("symbol")).agg(
                (pl.col("r").std().alias("volatility"))
            )
        ).collect()

        # Get the latest volatility for each symbol
        latest_volatilities = volatilities.select(
            pl.col("symbol"), "volatility"
        ).with_columns(pl.col("volatility").rank(method="dense", descending=True))

        picks: list[str] = []
        for symbol in view.symbols:
            if (
                symbol in daily_returns.columns
                and latest_volatilities.get_symbol(symbol, default=None)
                is not None
            ):
                volatility_rank = int(latest_volatilities.get_symbol(symbol, -1))
                if volatility_rank <= self._top_n:
                    picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

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