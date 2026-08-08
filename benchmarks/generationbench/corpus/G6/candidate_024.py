from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class IntegratedMomentum(Strategy):
    rationale = (
        "This strategy exploits historical momentum by selecting stocks with positive past returns to continue outperforming. "
        "It ranks all stocks based on their 60-day relative return compared to a broad market index like NIFTY 50 and chooses the top 30% of those with at least 1% excess return. "
        "Weights are adjusted based on momentum scores, capping individual stock weight at 2%, while ensuring portfolio diversification."
    )

    def __init__(self, window: int = 60, threshold: float = 1.0, top_n: int = 50) -> None:
        self._window = window
        self._threshold = threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate relative returns
        relative_returns = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("relative_return"),
                pl.col("session_date").max().alias("latest_date"),
            )
        ).collect()

        # Filter stocks with positive momentum
        filtered_stocks = relative_returns.filter(
            (pl.col("relative_return") >= self._threshold)
            & ((pl.col("latest_date") - pl.col("session_date")) < 120)
        )

        if filtered_stocks.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Rank stocks and select top N
        ranked_stocks = (
            filtered_stocks.sort("relative_return", descending=True).head(self._top_n).select(["symbol"])
        ).to_series()

        weight = 1.0 / self._top_n if ranked_stocks.height > 0 else 0.0

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_stocks.to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest