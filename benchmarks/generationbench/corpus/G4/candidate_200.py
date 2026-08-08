from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy exploits the phenomenon where significant price movements are "
        "often accompanied by increased trading volume. Large volume can signal strong "
        "conviction or liquidity, validating and sustaining directional trends."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_changes = {}
        price_directions = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            close_price = history.select(pl.col("adj_close")).select(
                pl.col(symbol).alias("close")
            ).to_series().to_list()
            adj_close_series = [float(p) for p in close_price]

            volume = history.select(pl.col("volume")).select(
                pl.col(symbol).alias("volume")
            ).to_series().to_list()
            volume_series = [int(v) for v in volume]

            if len(volume_series) < self._window:
                continue

            # Calculate daily percentage change in close price
            price_changes = [
                (close_price[i] - close_price[i - 1]) / close_price[i - 1]
                if close_price[i - 1] != 0 else 0.0 for i in range(1, len(close_price))
            ]
            # Calculate daily percentage change in volume
            vol_changes = [
                (volume_series[i] - volume_series[i - 1]) / volume_series[i - 1]
                if volume_series[i - 1] != 0 else 0.0 for i in range(1, len(volume_series))
            ]

            # Compute price direction as daily percentage change
            price_directions[symbol] = sum(price_changes) / self._window

            # Check for volume changes relative to the 20-day moving average of volume
            vol_avg = sum(volume_series[-self._window:]) / self._window
            if (vol_changes[-1] >= 0 and price_changes[-1] > 0) or \
               (vol_changes[-1] < 0 and price_changes[-1] < 0):
                volume_changes[symbol] = vol_avg

        # Rank stocks based on the magnitude of their price change and volume spike
        ranked_symbols = sorted(
            volume_changes.items(),
            key=lambda x: abs(price_directions[x[0]]) + abs(volume_changes[x[0]]),
            reverse=True,
        )

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / min(20, len(ranked_symbols))
        top_symbols = [s[0] for s in ranked_symbols[:min(20, len(ranked_symbols))]]

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest