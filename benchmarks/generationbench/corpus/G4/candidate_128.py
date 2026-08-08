from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMove(Strategy):
    rationale = (
        "This strategy exploits volume-confirmed directional moves by identifying stocks where "
        "significant price movements are accompanied by increased trading volume. These moves may "
        "indicate strong buying or selling interest and can lead to sustained trends."
    )

    def __init__(self, lookback_days: int = 5, volume_window: int = 20) -> None:
        self._lookback_days = lookback_days
        self._volume_window = volume_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days + self._volume_window - 1)

        if history.is_empty() or history.height < self._lookback_days + self._volume_window - 1:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[f"{symbol}.close"].to_list()]
            volumes = [float(v) for v in history[f"{symbol}.volume"].to_list()]

            if len(prices) < self._lookback_days + self._volume_window - 1 or \
                    len(volumes) < self._volume_window:
                continue

            price_changes = [(prices[i] / prices[0] - 1.0) for i in range(1, self._lookback_days)]
            volume_changes = [volumes[i] for i in range(self._volume_window)]

            if max(price_changes) > 0:  # Bullish signal
                avg_volume = sum(volume_changes) / len(volume_changes)
                if any(v > avg_volume * 1.5 for v in volume_changes):  # Strong volume increase
                    signals[symbol] = max(price_changes)

            elif min(price_changes) < 0:  # Bearish signal
                avg_volume = sum(volume_changes) / len(volume_changes)
                if any(v > avg_volume * 1.5 for v in volume_changes):  # Strong volume increase
                    signals[symbol] = -min(price_changes)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest