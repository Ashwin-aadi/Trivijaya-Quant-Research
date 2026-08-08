from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Trend following involves identifying the direction of a trend and riding it. "
        "Using volatility scaling can help manage risk while potentially capturing larger "
        "movements in price. This strategy aims to identify trending symbols based on their "
        "volatility over a lookback period."
    )

    def __init__(self, window: int = 20, volatility_window: int = 10) -> None:
        self._window = window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            latest_close = history.filter(pl.col("symbol") == symbol).select(
                pl.col("adj_close").last()
            ).item()

            # Calculate the volatility over the last `volatility_window` days
            vol_series = history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()[-self._volatility_window:]
            vol = (sum((v - latest_close for v in reversed(vol_series))) / len(vol_series)) ** 2

            # Determine the trend based on recent price action
            trend_score = sum(
                [1 if history.filter(pl.col("symbol") == symbol).select(pl.col("adj_close").head(i + 1)).to_numpy()[i][0] >= latest_close else -1 for i in range(self._window)]
            )

            # Scale the trend score by volatility
            scaled_trend = trend_score * vol ** 0.5

            symbol_trends[symbol] = scaled_trend

        # Filter out symbols with low or no trends
        picks = [symbol for symbol, score in symbol_trends.items() if abs(score) > 1]

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