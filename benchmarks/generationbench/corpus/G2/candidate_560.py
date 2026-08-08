from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large volume surges often precede price breaks, as institutional buying or selling "
        "can move the market. By identifying symbols with high volume and a directional price "
        "move, we can capitalize on these momentum events."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 20)
        if history.is_empty() or history.height < self._window + 20:
            return Signal(information_available_at=stamp, weights={})

        symbols_with_data = set(history["symbol"])
        history_df = (
            history
            .filter(pl.col("symbol").is_in(symbols_with_data))
            .sort("session_date")
        )
        
        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in symbols_with_data:
                continue
            
            prices = [float(v) for v in history_df.filter(pl.col("symbol") == symbol)["close"].to_list()]
            volumes = [float(v) for v in history_df.filter(pl.col("symbol") == symbol)["volume"].to_list()]

            # Check if the last price is a breakout
            latest_price = prices[-1]
            previous_prices = prices[:-1]

            # Calculate directional move
            max_price = max(previous_prices)
            min_price = min(previous_prices)

            if latest_price > max_price:
                direction = "up"
            elif latest_price < min_price:
                direction = "down"
            else:
                continue

            # Check for volume surge at the breakout point
            last_volume = volumes[-1]
            avg_volume = sum(volumes) / len(volumes)
            if last_volume > 1.5 * avg_volume and direction == "up":
                signals[symbol] = 0.2
            elif last_volume > 1.5 * avg_volume and direction == "down":
                signals[symbol] = -0.2

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        adjusted_weights = {s: w / total_weight for s, w in signals.items()}
        cash_allocation = 1.0 - sum(adjusted_weights.values())
        adjusted_weights.update({symbol: (w + cash_allocation) for symbol, w in adjusted_weights.items()})
        
        return Signal(
            information_available_at=stamp,
            weights={s: float(w) for s, w in adjusted_weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest