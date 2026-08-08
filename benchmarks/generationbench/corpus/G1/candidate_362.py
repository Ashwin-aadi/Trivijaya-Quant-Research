from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits trends by scaling the trend signal with volatility. "
        "During periods of high volatility, the strategy reduces its exposure to maintain stability."
    )

    def __init__(self, window: int = 20, lookback: int = 60) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        # Calculate log returns
        log_returns = (
            history
            .with_columns((pl.col("close").shift(-1) / pl.col("close") - 1).alias("log_return"))
            .sort("session_date", descending=False)
            .select(["symbol", "session_date", "log_return"])
        )

        # Calculate mean and std of log returns
        mean_log_return = (
            log_returns.groupby("symbol")
            .agg(
                (pl.col("log_return").mean()).alias("mean_log_return"),
                (pl.col("log_return").std()).alias("std_log_return"),
            )
        ).collect()

        # Filter out symbols with insufficient data
        if mean_log_return.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Compute volatility-scaled trend signal
        symbol_list = view.symbols
        weights: dict[str, float] = {}
        for symbol in symbol_list:
            latest_close = view.latest_close()[symbol]
            mean_log_return_val = mean_log_return.filter(pl.col("symbol") == symbol).select("mean_log_return").to_series().item()
            std_log_return_val = mean_log_return.filter(pl.col("symbol") == symbol).select("std_log_return").to_series().item()

            if std_log_return_val > 0:
                trend_signal = (latest_close - latest_close * mean_log_return_val) / std_log_return_val
                weights[symbol] = trend_signal

        # Normalize weights to ensure they sum up to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in normalized_weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest