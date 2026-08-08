from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum36m(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum in the Indian market by identifying "
        "stocks with positive returns over a recent period and allocating positions based on their "
        "performance. High-momentum stocks are expected to continue outperforming due to investor "
        "behavior biases and information delays."
    )

    def __init__(self, lookback_days: int = 180, top_n: int = 25) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate cumulative returns for each stock
        returns = (
            history.group_by("symbol")
                   .agg(
                       (pl.col("close") / pl.col("open").shift(self._lookback_days) - 1.0).alias("cumulative_return"),
                   )
        ).sort("cumulative_return", descending=True)

        if returns.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_stocks = [row["symbol"] for row in returns.to_dicts()[:self._top_n]]

        # Compute weight for each selected stock
        top_returns = history.select([pl.col("symbol"), pl.col("close").alias("latest_close")]).filter(pl.col("symbol").is_in(top_stocks))
        weights = {s: float(top_returns.filter(pl.col("symbol") == s)["latest_close"].item()) / sum(top_returns["latest_close"].to_list()) for s in top_stocks}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest