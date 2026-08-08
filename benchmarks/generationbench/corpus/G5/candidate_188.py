from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are likely to continue in the near term. "
        "This strategy identifies stocks that have made a significant upward or downward move "
        "in price while also seeing an increase in trading volume."
    )

    def __init__(self, window: int = 20, threshold_price_change: float = 0.03, min_volume_increase: float = 1.1) -> None:
        self._window = window
        self._threshold_price_change = threshold_price_change
        self._min_volume_increase = min_volume_increase

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        moves: dict[str, tuple[float, float]] = {}
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol)
            opens = [float(o) for o in df["open"].to_list()]
            closes = [float(c) for c in df["close"].drop_nulls().to_list()]
            volumes = [float(v) for v in df["volume"].to_list()]

            if len(opens) < self._window or len(closes) < self._window:
                continue

            last_close = closes[-1]
            price_change = (last_close - opens[0]) / opens[0]

            if abs(price_change) < self._threshold_price_change:
                continue

            volume_change = volumes[-1] / volumes[0]
            if not (self._min_volume_increase <= volume_change <= 2.0):
                continue

            moves[symbol] = (price_change, volume_change)

        if not moves:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(moves)
        selected_symbols = [s for s in moves.keys()]
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).collect().row(0)[0]
    assert isinstance(newest, date)
    return newest