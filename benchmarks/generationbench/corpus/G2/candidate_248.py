from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment in a particular "
        "direction. If a stock has high volume on a significant price move, it suggests that the "
        "move is likely to continue or that there might be buying pressure, which can generate returns."
    )

    def __init__(self, window: int = 10, threshold_multiplier: float = 2.0) -> None:
        self._window = window
        self._threshold_multiplier = threshold_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes().drop_nulls().to_dict(as_series=False)
        symbols = list(latest_closes.keys())

        price_changes = {}
        for symbol in symbols:
            close_values = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "adj_close"].to_list()]
            if len(close_values) < self._window + 1:
                continue

            latest_price = float(latest_closes[symbol])
            previous_prices = close_values[:-1]

            # Calculate the price change and volume in the last period
            latest_price_change = (latest_price - previous_prices[-1]) / previous_prices[
                -1]
            latest_volume = int(history.filter(pl.col("symbol") == symbol)[
                                    "volume"].sum())

            if len(previous_prices) < self._window:
                continue

            # Calculate the average price change over the last window days
            avg_price_change = sum([
                (close_values[i + 1] - close_values[i]) / close_values[i]
                for i in range(len(close_values) - 1)]) / len(close_values) - 1

            threshold = abs(avg_price_change) * self._threshold_multiplier

            if latest_price_change > threshold:
                price_changes[symbol] = (latest_price, latest_volume)
            elif latest_price_change < -threshold:
                price_changes[symbol] = (latest_price, latest_volume)

        if not price_changes:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(price_changes)
        signal_weights = {symbol: weight for symbol in price_changes.keys()}
        return Signal(
            information_available_at=stamp,
            weights=signal_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest