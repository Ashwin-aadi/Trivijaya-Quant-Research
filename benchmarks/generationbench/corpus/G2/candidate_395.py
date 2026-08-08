from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy seeks to capture returns by identifying stocks that show both strong "
        "volume and positive momentum. High volume suggests liquidity and possibly higher "
        "trading activity, while positive momentum indicates recent price appreciation."
    )

    def __init__(self, momentum_window: int = 10, volume_threshold: float = 1e6) -> None:
        self._momentum_window = momentum_window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + 1)

        if history.is_empty() or history.height < self._momentum_window + 1:
            return Signal(information_available_at=stamp, weights={})

        momentum: dict[str, float] = {}
        volume: dict[str, float] = {}

        for symbol in view.symbols:
            closes = [float(v) for v in history.select("adj_close", pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(closes) < self._momentum_window + 1:
                continue

            # Calculate momentum
            latest_close = float(history.filter(pl.col("symbol") == symbol).select(pl.last("adj_close")).item())
            momentum_change = (latest_close - closes[0]) / max(1, abs(closes[0] - closes[-1]))
            momentum[symbol] = momentum_change

            # Calculate volume
            volumes = [float(v) for v in history.select("volume", pl.col("symbol") == symbol)["volume"].to_list()]
            recent_volume = sum(volumes[-self._momentum_window:])
            volume[symbol] = recent_volume / self._momentum_window if recent_volume > 0 else 0

        # Filter by volume
        filtered_symbols = [s for s in momentum.keys() if volume[s] >= self._volume_threshold]

        # If no symbols meet the criteria, return an empty Signal
        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest