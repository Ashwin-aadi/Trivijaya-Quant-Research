from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong momentum and can suggest future "
        "price action. If a stock has a large volume increase in the direction of its price move, "
        "it suggests that institutional or retail investors are actively participating in the trend."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols that do not have enough data
        symbols_with_data = [s for s in view.symbols if s in history.columns]
        if len(symbols_with_data) < 5:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the percentage change and volume change for each symbol
        changes = (
            history[symbols_with_data]
            .with_columns(
                (pl.col("close") - pl.col("adj_close").shift(1)) / pl.col("adj_close").shift(1).alias("price_change"),
                (pl.col("volume") - pl.col("volume").shift(1)).alias("volume_change"),
            )
            .sort("session_date", descending=False)
        )

        # Select the last valid price and volume change
        latest_changes = changes.select(
            *[f"{symbol}_pc" for symbol in symbols_with_data],
            *[f"{symbol}_vc" for symbol in symbols_with_data]
        ).to_dict(False)

        # Filter out symbols with insignificant moves or no volume increase
        active_symbols = []
        for symbol, data in latest_changes.items():
            price_change = float(data[1])  # The first element is the current session's change
            volume_change = float(data[2])  # The second element is the previous session's change

            if abs(price_change) > 0.05 and volume_change > 0:  # Arbitrary thresholds for significance
                active_symbols.append(symbol)

        if not active_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(active_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in active_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest