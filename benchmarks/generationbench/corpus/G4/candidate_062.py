from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the persistence in asset price trends during periods of low "
        "volatility. By scaling our position size relative to recent volatility levels, we can "
        "capture trends efficiently while managing risk more conservatively in volatile markets."
    )

    def __init__(self, trend_window: int = 100, vol_window: int = 20, rank_threshold: float = -2.0) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window
        self._rank_threshold = rank_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._vol_window + self._trend_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        volatility_df = _calculate_volatility(history)
        trend_df = _calculate_trend(view.closes(lookback=self._trend_window))

        combined_df = (
            pl.concat([volatility_df, trend_df], how="horizontal")
                .with_columns(pl.col("symbol").cast(pl.Categorical))
                .group_by("symbol")
                .agg(
                    [
                        (pl.col("volatility") < self._rank_threshold).alias("low_vol"),
                        (pl.col("trend") == 1.0).alias("up_trend"),
                        pl.count().alias("count"),
                    ]
                )
        )

        ranked_df = (
            combined_df.with_columns(
                ((pl.col("low_vol") & pl.col("up_trend")) * pl.col("count")).rank(method="dense", descending=True)
            )
            .select(["symbol", "rank"])
            .sort("rank")
            .to_dict(False)
        )

        if not ranked_df:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in ranked_df[:10]]
        weight = 0.1 / len(top_symbols)  # Assuming a maximum of 10 symbols and a total allocation of 10%
        signal = {symbol: weight for symbol in top_symbols}
        return Signal(information_available_at=stamp, weights=signal)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(history: pl.DataFrame) -> pl.DataFrame:
    open_close_diff = (history["close"] / history["open"].shift(1) - 1.0).alias("return")
    volatility = (
        (pl.col("return").rolling_std(window_size=self._vol_window, center=False)).mean().alias("volatility")
    )
    return (
        history
            .with_column(volatility)
            .select(["symbol", "session_date", "adj_close", "volatility"])
    )


def _calculate_trend(closes: pl.DataFrame) -> pl.DataFrame:
    returns = (closes["adj_close"] / closes["adj_close"].shift(1) - 1.0).alias("return")
    trend = (pl.col("return").rolling_mean(window_size=self._trend_window, center=False)).rank(method="dense", descending=True)
    return (
        closes
            .with_column(trend.cast(pl.Int64))
            .select(["symbol", "session_date", "adj_close", "trend"])
    )