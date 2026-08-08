from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy focuses on exploiting high-volatility periods that precede significant trends. "
        "By scaling positions based on volatility and identifying trend signals, the strategy aims to maximize returns while managing risk."
    )

    def __init__(self, window: int = 20, ma_short: int = 50, ma_long: int = 200) -> None:
        self._window = window
        self._ma_short = ma_short
        self._ma_long = ma_long

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute realized volatility
        prices = history.select(pl.col("close")).to_pandas()
        returns = prices.pct_change().dropna() * 100
        volatilities = returns.rolling_std(window=self._window)

        # Identify trend using moving average crossover
        ma_short = (
            history.select(pl.col("adj_close"))
                   .rolling_mean(window=self._ma_short)
                   .to_series()
        )
        ma_long = (
            history.select(pl.col("adj_close"))
                   .rolling_mean(window=self._ma_long)
                   .to_series()
        )

        trend_signals = (ma_short - ma_long).to_numpy()

        # Rank symbols by volatility and trend signal
        rankings = []
        for symbol in view.symbols:
            if f"close_{symbol}" not in prices.columns or \
               f"volatility_{symbol}" not in volatilities.columns or \
               f"trend_signal_{symbol}" not in trend_signals:
                continue

            close_values = [float(v) for v in history[history["symbol"] == symbol]["adj_close"].drop_nulls().to_list()]
            volatility = float(volatilities[f"volatility_{symbol}"].max())
            trend_signal = trend_signals[len(trend_signals) - 1][list(prices.columns).index(f"close_{symbol}")]

            if len(close_values) >= self._window:
                rank_score = (trend_signal + volatility) / 2
                rankings.append((symbol, rank_score))

        # Select top symbols based on ranking
        rankings.sort(key=lambda x: x[1], reverse=True)
        picks = [rank[0] for rank in rankings[:20]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
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