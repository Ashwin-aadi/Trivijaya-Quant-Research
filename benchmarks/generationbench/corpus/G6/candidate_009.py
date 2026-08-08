from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy identifies and invests in stocks with lower historical volatility to "
        "ensure portfolio stability. By focusing on low-volatility equities, the approach aims "
        "to manage risk effectively while seeking consistent performance."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5, min_capitalization: float = 1e9) -> None:
        self._window = window
        self._threshold = threshold
        self._min_capitalization = min_capitalization

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.select(pl.col("symbol"), (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"))
            .sort("session_date")
            .drop_nulls()
            .group_by("symbol")
            .agg((pl.col("r").std().alias("volatility")))
        )

        # Filter by minimum capitalization
        market_caps = view.closes(lookback=self._window)
        filtered_symbols = [sym for sym in returns.columns if float(view.latest_close()[sym]) >= self._min_capitalization]
        filtered_returns = returns.select(pl.col("symbol").filter(pl.col("symbol").is_in(filtered_symbols)), "volatility")

        # Rank symbols by volatility
        ranked_volatility = (
            filtered_returns.sort("volatility")
            .group_by("symbol")
            .agg((pl.col("volatility").rank(method="dense", descending=False).alias("rank")))
        )

        # Select bottom 30% of symbols based on rank and apply exit rule
        bottom_30_percent = int(len(filtered_symbols) * 0.3)
        selected_symbols = [row[0] for row in ranked_volatility.rows()[:bottom_30_percent]]

        # Rebalance every month, exit if volatility exceeds threshold
        current_month = stamp.month
        exit_condition = False
        for symbol in selected_symbols:
            recent_volatility = float(filtered_returns.filter(pl.col("symbol") == symbol)["volatility"][0])
            if recent_volatility > self._threshold * filtered_returns.select(pl.col("volatility").median())[0][0]:
                exit_condition = True
                break

        if not selected_symbols or exit_condition:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest