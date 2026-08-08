from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy leverages volatility-scaled trend-following to capitalize on persistent "
        "trends during low-volatility periods. It adjusts position sizes based on recent historical "
        "volatility, seeking to mitigate risks in high-volatility phases while benefiting from trends."
    )

    def __init__(self, short_window: int = 50, long_window: int = 200, volatility_window: int = 20, max_positions: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._volatility_window = volatility_window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._short_window, self._long_window, self._volatility_window))
        
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]

        # Compute moving averages
        ma_short = (
            history.with_columns(
                (pl.col("adj_close").rolling_mean(window_size=self._short_window).alias(f"ma_{self._short_window}"))
            )
            .group_by("symbol")
            .agg((pl.col(f"ma_{self._short_window}").last().alias("ma_short")))
        )

        ma_long = (
            history.with_columns(
                (pl.col("adj_close").rolling_mean(window_size=self._long_window).alias(f"ma_{self._long_window}"))
            )
            .group_by("symbol")
            .agg((pl.col(f"ma_{self._long_window}").last().alias("ma_long")))
        )

        # Calculate trend signal
        trend_signal = (
            ma_short.join(ma_long, on="symbol", how="inner")
            .with_columns(
                (pl.col(f"ma_{self._short_window}") - pl.col(f"ma_{self._long_window}")).alias("trend_signal")
            )
            .sort("trend_signal", descending=True)
        )

        # Calculate volatility
        log_returns = (
            history.with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("log_return"))
            .group_by("symbol")
            .agg(pl.col("log_return").std().alias("volatility"))
        )

        # Combine trend signal and volatility
        combined = (
            trend_signal.join(log_returns, on="symbol", how="inner")
            .sort(f"trend_signal/{f'volatility_{self._volatility_window}'}/desc")
        )

        # Select top N symbols based on the combination of trend strength and low volatility
        picks: list[str] = [row["symbol"] for row in combined.to_dict() if len(picks) < self._max_positions]

        weight = 1.0 / len(picks)
        return Signal(information_available_at=stamp, weights={s: weight for s in picks})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest