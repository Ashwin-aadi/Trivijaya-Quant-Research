from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedBreakout(Strategy):
    rationale = (
        "This strategy identifies and leverages volume-confirmed directional moves in the Indian market. "
        "It looks for price breakouts confirmed by increased trading volumes to enter trades with high confidence."
    )

    def __init__(self, window: int = 10, breakout_threshold: float = 0.1, vol_confirmed_threshold: float = 0.25) -> None:
        self._window = window
        self._breakout_threshold = breakout_threshold
        self._vol_confirmed_threshold = vol_confirmed_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        latest_closes = {symbol: float(view.latest_close()[symbol]) for symbol in symbols}

        signals = []
        for symbol in symbols:
            hist_df = history.select(
                pl.col("session_date"), pl.col(symbol).alias("close")
            ).to_pandas()

            if len(hist_df) < self._window + 2:
                continue

            prices = hist_df["close"].tolist()
            volumes = view.history(lookback=self._window + 2).select(
                pl.col("session_date"), pl.col(symbol).alias("volume")
            ).to_pandas()["volume"].tolist()

            recent_close = latest_closes[symbol]
            recent_volume = view.latest_close()[symbol]

            # Calculate price breakout
            if (
                prices[-3] < prices[-2] * (1 + self._breakout_threshold)
                and prices[-2] >= prices[-1] * (1 + self._breakout_threshold)
                and prices[-2] > recent_close * (1 + self._breakout_threshold / 2)
            ):
                # Check volume confirmation
                vol_avg_recent = sum(volumes[-self._window:]) / self._window
                if volumes[-1] >= vol_avg_recent * (1 + self._vol_confirmed_threshold):
                    signals.append(symbol)

        weights = {s: 0.05 for s in signals} if signals else {}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest