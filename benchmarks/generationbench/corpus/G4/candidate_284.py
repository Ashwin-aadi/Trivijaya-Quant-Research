from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedTrend(Strategy):
    rationale = (
        "This strategy exploits volume-confirmed directional moves by identifying instances "
        "where significant trading volume aligns with a clear price trend. Large institutional "
        "or high-volume traders often drive prices in one direction, and their trades are "
        "confirmed by increased trading volume, signaling strong investor confidence or reaction."
    )

    def __init__(self, trend_window: int = 20, volume_threshold: float = 1.5) -> None:
        self._trend_window = trend_window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._trend_window + 30)
        if closes.height < self._trend_window + 30:
            return Signal(information_available_at=stamp, weights={})

        trend_strengths: dict[str, float] = {}
        volume_participation_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._trend_window + 30:
                continue

            # Calculate the trend strength using 20-day SMA
            sma_20 = sum(values[-self._trend_window:]) / self._trend_window
            current_close = values[-1]
            trend_strengths[symbol] = current_close - sma_20

            # Calculate the volume participation score
            history = view.history(lookback=self._trend_window + 30)
            volumes = [float(v) for v in history["volume"][symbol].drop_nulls().to_list()]
            avg_volume = sum(volumes[-self._trend_window:]) / self._trend_window
            today_volume = values[1]  # Close value at the end of the day
            if today_volume >= self._volume_threshold * avg_volume:
                volume_participation_scores[symbol] = 1.0
            else:
                volume_participation_scores[symbol] = 0.0

        candidates: list[str] = [
            s for s in trend_strengths.keys() if s in volume_participation_scores and volume_participation_scores[s]
        ]
        top_n_candidates = candidates[:20]

        if not top_n_candidates:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to each selected stock
        weights = {symbol: 1.0 / len(top_n_candidates) for symbol in top_n_candidates}
        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest