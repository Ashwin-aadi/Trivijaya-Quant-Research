from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following seeks to capitalize on trending behavior while "
        "limiting exposure during periods of high volatility. High-volatility regimes are "
        "associated with higher risks and costs, so reducing exposure in such times can "
        "lead to better risk-adjusted returns."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the log returns
        log_returns = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("r")
        history_with_returns = history.with_columns(log_returns)
        
        # Calculate volatility using the standard deviation of returns
        vol = (
            history_with_returns.group_by("symbol")
            .agg(pl.col("r").std().alias("volatility"))
            .collect()
        )

        # Get recent closing prices to compare with historical trends
        closes = view.closes(lookback=self._window).fill_null(0.0)
        
        # Compute the trend signal as the product of price change and volatility
        trend_signal = (
            history_with_returns.join(
                vol, on="symbol", how="left"
            )
            .with_columns(
                (pl.col("adj_close") - pl.col("adj_close").shift(self._window)).abs()
                / pl.col("volatility")
                .rank(method="dense", descending=True)
                .alias("trend_signal")
            )
        )

        # Identify symbols with strong trend signals
        top_symbols = (
            trend_signal.sort("trend_signal", descending=True)["symbol"]
            .to_list()[:5]
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest