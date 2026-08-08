from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy leverages cross-sectional momentum by selecting stocks with high "
        "momentum scores based on percentage change and cumulative returns over 20 days. It "
        "ensures a portfolio of top performers while maintaining risk through dynamic entry "
        "and exit rules."
    )

    def __init__(self, window: int = 20, momentum_threshold: float = 0.3, loss_bound: float = 5.0) -> None:
        self._window = window
        self._momentum_threshold = momentum_threshold
        self._loss_bound = loss_bound

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate percentage change and cumulative returns
        history = (
            history
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("percentage_change")
            )
            .with_columns(
                pl.col("adj_close").rolling_sum(window=self._window, closed="both").over("symbol").alias("cumulative_return")
            )
        )

        # Filter to keep only the most recent window period
        history = history.sort("session_date", descending=True).head(self._window)

        # Calculate momentum score
        momentum_scores = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("percentage_change").mean().alias("avg_percentage_change")),
                (pl.col("cumulative_return")[-1] / pl.col("adj_close")[0]).alias("cumulative_return_ratio"),
            )
        )

        # Rank symbols based on momentum scores
        ranked_symbols = (
            momentum_scores
            .sort(
                pl.col("avg_percentage_change").rank(method="ordinal", descending=True),
                pl.col("cumulative_return_ratio").rank(method="ordinal", descending=True)
            )
            .select(pl.col("symbol"))
        )

        # Select top 30% of symbols based on ranking
        num_symbols = len(ranked_symbols)
        top_n = int(num_symbols * self._momentum_threshold)
        selected_symbols = ranked_symbols.slice(0, top_n).to_list()[0]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate equal weights for selected symbols
        weight = 1.0 / len(selected_symbols)
        signal_weights = {symbol: weight for symbol in selected_symbols}

        return Signal(
            information_available_at=stamp,
            weights=signal_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest