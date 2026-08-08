from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "The strategy aims to capture significant trends by scaling positions based on recent "
        "market volatility. High volatility suggests increased risk and caution in position sizing; "
        "low volatility indicates smoother price movements, allowing for larger positions."
    )

    def __init__(self, ma_window: int = 50, vol_threshold_percentile: float = 15, exit_volatility_percentile: float = 75, max_positions: int = 30) -> None:
        self._ma_window = ma_window
        self._vol_threshold_percentile = vol_threshold_percentile
        self._exit_volatility_percentile = exit_volatility_percentile
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=20 + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.latest_close()

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )

        # Calculate moving average
        ma_column = f"ma_{self._ma_window}"
        history = history.with_columns(
            (pl.col("adj_close").rolling_mean(self._ma_window)).alias(ma_column)
        )

        # Calculate volatility as rolling standard deviation of daily returns
        vol_column = "volatility"
        history = history.with_columns(
            (pl.col("return").rolling_std(20).alias(vol_column))
        )

        # Identify signals based on the strategy's criteria
        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in latest_closes or symbol not in history.columns:
                continue

            close_value = float(latest_closes[symbol])
            ma_value = float(history.filter(pl.col("symbol") == symbol)[ma_column].last())
            vol_value = float(history.filter(pl.col("symbol") == symbol)[vol_column].last())

            if (close_value > ma_value and
                    pl.col(vol_column).quantile(self._vol_threshold_percentile / 100) >= vol_value):
                signals.append(symbol)
            elif (close_value < ma_value and
                  pl.col(vol_column).quantile((100 - self._exit_volatility_percentile) / 100) <= vol_value):
                signals.append(symbol)

        if len(signals) > self._max_positions:
            signals = signals[:self._max_positions]

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest