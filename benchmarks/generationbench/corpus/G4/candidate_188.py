from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMove(Strategy):
    rationale = (
        "This strategy exploits volume-confirmed directional moves by identifying significant price movements "
        "accompanied by increased trading volume. High volume often indicates stronger trends and confirms the direction of the move."
    )

    def __init__(self, window: int = 20, threshold_price_change: float = 0.01, threshold_volume_change: float = 1.5) -> None:
        self._window = window
        self._threshold_price_change = threshold_price_change
        self._threshold_volume_change = threshold_volume_change

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            row = history.filter(pl.col("symbol") == symbol).sort("session_date").tail(2)
            open_val = float(row["open"][0])
            close_val = float(row["close"][-1])
            high_val = float(row["high"][-1])
            low_val = float(row["low"][-1])
            volume = int(row["volume"][-1])
            prev_volume = int(row["volume"][-2])

            price_change = (close_val - open_val) / open_val
            if abs(price_change) < self._threshold_price_change:
                continue

            avg_volume = history.filter(pl.col("symbol") == symbol)["volume"].mean().to_list()[0]
            volume_change = volume - prev_volume
            score = price_change * (volume_change / avg_volume)

            scores[symbol] = score

        sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        picks = list(sorted_scores.keys())[:20]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest