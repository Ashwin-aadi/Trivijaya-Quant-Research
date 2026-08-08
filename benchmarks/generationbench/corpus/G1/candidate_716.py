from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong conviction among market participants. "
        "We look for significant price movements accompanied by high volume to identify potential"
        "trend changes or continuations."
    )

    def __init__(self, window: int = 10, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)

        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "volume" not in history.columns:
                continue

            price_changes = [float(v) for v in history[symbol].to_list()]
            volumes = [float(v) for v in history["volume"].drop_nulls().to_list()]

            # Calculate the change in price and volume
            recent_close = price_changes[-1]
            last_close = price_changes[-2]

            if (recent_close - last_close) / abs(last_close) >= self._threshold:
                if volumes[-1] > 1.5 * max(volumes[:-1]):
                    signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest