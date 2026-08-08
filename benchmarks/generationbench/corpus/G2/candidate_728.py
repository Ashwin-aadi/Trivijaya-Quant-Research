from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are a signal that institutional buyers or sellers "
        "are entering the market. High volume with a price movement in the same direction can "
        "indicate strong conviction and potentially sustained momentum."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols with insufficient history
        valid_symbols = [symbol for symbol in view.symbols if symbol in history.symbol.unique()]

        signals: dict[str, float] = {}
        for symbol in valid_symbols:
            df = history.filter(pl.col("symbol") == symbol)
            last_close = float(df.select("adj_close").tail(1)["adj_close"][0])
            closes = [float(v) for v in df.select("adj_close").drop_nulls().to_list()]
            volumes = [float(v) for v in df.select("volume").drop_nulls().to_list()]

            if len(closes) < self._window + 2:
                continue

            # Check the direction of movement
            last_move_direction = (last_close - closes[-2]) / abs(closes[-2])
            volume_change = volumes[-1] - volumes[-2]

            # Confirm with volume
            if abs(volume_change) > max(volumes[-self._window:]) * 0.5 and last_move_direction != 0:
                signals[symbol] = 1.0 / len(signals)

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest