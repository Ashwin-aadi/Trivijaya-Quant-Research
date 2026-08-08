from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySMA(Strategy):
    rationale = (
        "This strategy captures trends in stock prices while adjusting exposure based on "
        "prevailing volatility levels. It uses a 50-day Simple Moving Average (SMA) crossing "
        "above or below the 200-day SMA as entry signals and exits when volatility increases "
        "significantly or price retraces by more than 50%."
    )

    def __init__(self, window_sma_50: int = 50, window_sma_200: int = 200, atr_window: int = 14, lookback_volatility: int = 20) -> None:
        self._window_sma_50 = window_sma_50
        self._window_sma_200 = window_sma_200
        self._atr_window = atr_window
        self._lookback_volatility = lookback_volatility

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_sma_200 + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()

        sma_50 = history.select(
            pl.col("symbol"),
            (pl.col("close").shift(-self._window_sma_50).rolling_mean(self._window_sma_50)).alias(f"sma_{self._window_sma_50}"),
        )
        sma_200 = history.select(
            pl.col("symbol"),
            (pl.col("close").shift(-self._window_sma_200).rolling_mean(self._window_sma_200)).alias(f"sma_{self._window_sma_200}"),
        )
        sma_crosses = sma_50.join(sma_200, on="symbol").select(
            pl.col("symbol"), (pl.col(f"sma_{self._window_sma_50}") > pl.col(f"sma_{self._window_sma_200}")).alias("cross_above")
        )

        atr = history.select(
            pl.col("symbol"),
            (pl.col("high").shift(-1) - pl.col("low").shift(-1)).abs().rolling_mean(self._atr_window).alias(f"atr_{self._atr_window}"),
        )
        mean_return_20d = closes.sort("session_date", descending=True).head(self._lookback_volatility + 1).select(
            pl.col("symbol"), (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
        ).group_by("symbol").agg(pl.col("returns").mean().alias(f"mean_return_20d"))
        
        volatility_condition = atr.join(mean_return_20d, on="symbol").select(
            pl.col("symbol"),
            (pl.col(f"atr_{self._atr_window}") > 1).alias("high_volatility"),
            ((pl.col("close") / pl.col("adj_close").shift(1) - 1.0) < -0.5).alias("retrace_50%"),
        )

        def get_signal(row):
            if row["cross_above"] and not (row["high_volatility"] or row["retrace_50%"]):
                return "buy"
            elif not row["cross_above"] and row["high_volatility"] or row["retrace_50%"]:
                return "sell"
            else:
                return None

        signals = sma_crosses.join(volatility_condition, on="symbol").with_columns(get_signal().alias("signal"))

        buys = [s for s in signals.filter(pl.col("signal") == "buy").get_column("symbol").to_list() if s in closes.columns]
        sells = [s for s in signals.filter(pl.col("signal") == "sell").get_column("symbol").to_list() if s in closes.columns]

        weight_buy = 1.0 / len(buys) if buys else 0
        weight_sell = -1.0 / len(sells) if sells else 0

        weights = {s: weight_buy for s in buys}
        weights.update({s: weight_sell for s in sells})

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest