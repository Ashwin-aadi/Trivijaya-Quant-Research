from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits mean-reverting behavior within volatile periods by "
        "scaling trend following signals based on current market volatility. Higher volatility "
        "periods reduce trade sizes to manage risk effectively while capturing potential profits."
    )

    def __init__(self, window: int = 20, sma_window: int = 50, scaling_factor_min: float = 0.1, scaling_factor_max: float = 1.0) -> None:
        self._window = window
        self._sma_window = sma_window
        self._scaling_factor_min = scaling_factor_min
        self._scaling_factor_max = scaling_factor_max

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        daily_returns = (closes[closes.columns[1:]] / closes[closes.columns[:-1]].shift(1) - 1.0).fillna(pl.Series([0]))

        # Compute rolling volatility
        vol_measure = daily_returns.std().alias("volatility")
        vol_series = daily_returns.with_columns(vol_measure).select("volatility")

        # Calculate 50-day simple moving average (SMA)
        sma = view.closes(lookback=self._sma_window)[["symbol", "close"]].join(
            vol_series, on="symbol", how="inner"
        ).with_column(
            pl.col("close").rolling_mean(self._sma_window).alias("sma")
        )

        # Identify upward and downward trends
        trends = sma.with_columns(
            (pl.col("close") > pl.col("sma")).alias("upward_trend"),
            (pl.col("close") < pl.col("sma")).alias("downward_trend")
        )

        # Scale trade size based on volatility
        scaling_factor = (
            self._scaling_factor_min + (self._scaling_factor_max - self._scaling_factor_min) * (1.0 / (1.0 + vol_series["volatility"].max()))
        )
        trends = trends.with_column(scaling_factor.alias("scaling_factor"))

        # Rank candidates based on trend signals and volatility
        picks: list[str] = []
        for symbol in view.symbols:
            upward_trend = trends.filter(pl.col("symbol") == symbol).select("upward_trend").to_series().to_list()[0]
            downward_trend = trends.filter(pl.col("symbol") == symbol).select("downward_trend").to_series().to_list()[0]
            volatility = vol_series.filter(pl.col("symbol") == symbol).select("volatility").to_series().to_list()[0]

            if upward_trend and volatility < 0.2:
                picks.append(symbol)
            elif downward_trend and volatility > 0.8:
                picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = scaling_factor / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest