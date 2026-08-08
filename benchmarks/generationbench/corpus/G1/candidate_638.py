from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollow(Strategy):
    rationale = (
        "This strategy follows trends that have a high volatility. High volatility indicates "
        "that the asset is currently experiencing significant price movements, which we can use "
        "to our advantage by entering positions in the direction of the trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility_trend_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol)
            prices = [float(v) for v in df["adj_close"].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            # Calculate the log returns
            log_returns = [(prices[i] / prices[i - 1] - 1.0) for i in range(1, len(prices))]

            # Calculate volatility as the standard deviation of log returns
            volatility = (pl.Series(log_returns).std()) * (252 ** 0.5)
            if not pl.is_nan(volatility):
                # Normalize by mean log return to get a trend score
                mean_log_return = sum(log_returns) / len(log_returns)
                trend_score = (mean_log_return + volatility) / (1 - mean_log_return)

                volatility_trend_scores[symbol] = float(trend_score)

        if not volatility_trend_scores:
            return Signal(information_available_at=stamp, weights={})

        # Select the top symbol with the highest volatility-trend score
        top_symbol = max(volatility_trend_scores.items(), key=lambda x: x[1])[0]
        weight = 1.0

        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest