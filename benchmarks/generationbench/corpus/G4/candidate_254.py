from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the relationship between market trends and volatility in "
        "the Indian equity markets. By scaling trades based on recent volatility, it aims to "
        "capitalize on prolonged trending behavior during low-volatility periods."
    )

    def __init__(self, window: int = 20, sma_window: int = 50) -> None:
        self._window = window
        self._sma_window = sma_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._sma_window)
        if history.is_empty() or history.height < self._window + self._sma_window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = view.closes(lookback=self._window).select(
            pl.all().exclude("session_date").to_series()
        ).transpose().hstack([pl.col("session_date")])

        # Calculate log returns
        returns = (
            history.select(pl.col("symbol"), pl.col("adj_close"))
                   .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("log_return"))
                   .group_by("symbol")
                   .agg((pl.col("log_return").std().alias("volatility")))
        )

        # Calculate 50-day SMA
        sma = (
            history.select(pl.col("symbol"), pl.col("close"))
                   .with_columns((pl.col("close").rolling_mean(window_size=self._sma_window).alias(f"sma_{self._sma_window}")))
                   .group_by("symbol")
                   .agg([(f"sma_{self._sma_window}", "last")])
        )

        # Join returns and SMA
        combined = closes.join(returns, on="symbol").join(sma, on="symbol")

        # Determine position sizing based on volatility
        vol_quantiles = (
            combined.select(pl.col("volatility"))
                   .quantile([0.2, 0.8], interpolation="linear")
        )
        v20 = float(vol_quantiles[0])
        v80 = float(vol_quantiles[1])

        # Determine trend direction
        trends = []
        for symbol in symbols:
            close_price = combined.select(f"{symbol}").item()
            sma_value = combined.select(f"sma_{self._sma_window}").item()
            if close_price > sma_value:
                trends.append("bullish")
            else:
                trends.append("bearish")

        # Position sizing
        weight = 1.0 / len(symbols)
        leverage = {symbol: weight for symbol in symbols}
        for i, symbol in enumerate(symbols):
            if combined.select(f"{symbol}").item() > v20 and combined.select(f"volatility").item() < v20:
                leverage[symbol] *= 1.5
            elif v20 <= combined.select(f"volatility").item() < v80:
                continue
            else:
                leverage[symbol] /= 2

        # Apply risk limits and ensure total exposure does not exceed 30 stocks
        if len(leverage) > 30:
            top_30_symbols = sorted(leverage.items(), key=lambda x: abs(x[1]), reverse=True)[:30]
            leverage = {symbol: weight for symbol, weight in top_30_symbols}

        # Ensure no single stock dominates the portfolio
        max_weight = max(leverage.values())
        if any(weight > 5.0 / 100 for weight in leverage.values()):
            for symbol in symbols:
                if leverage[symbol] > 5.0 / 100 and len(leverage) < 30:
                    continue
                else:
                    leverage[symbol] = max_weight

        return Signal(
            information_available_at=stamp, weights={s: weight for s, weight in leverage.items() if weight > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest