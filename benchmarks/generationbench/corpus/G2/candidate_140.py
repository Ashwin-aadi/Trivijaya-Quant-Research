from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves exploit the idea that high volume on a price "
        "movement indicates institutional participation and strength in the move. This "
        "strategy captures stocks with significant volume and price movement."
    )

    def __init__(self, window: int = 20, min_volume_factor: float = 1.5) -> None:
        self._window = window
        self._min_volume_factor = min_volume_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol)
            opens = [float(o) for o in df.select("open").to_series().to_list()[0]]
            closes = [float(c) for c in df.select("close").to_list()[0]]

            if len(opens) < self._window:
                continue

            price_move = max(closes[-1] - min(closes), 0)
            volume_move = (df.select(pl.col("volume").max() / df.select(pl.col("volume").min())).item()
                           - 1.0)

            if price_move > 0 and volume_move >= self._min_volume_factor:
                signals.append(symbol)

        weight = 1.0 / len(signals) if signals else 0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest