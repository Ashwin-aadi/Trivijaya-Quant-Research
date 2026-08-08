from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy identifies strong buying or selling pressure by combining directional "
        "price movements with significant trading volumes. High volume during price trends "
        "suggests a stronger move and enhances the reliability of trend signals."
    )

    def __init__(self, window: int = 5, threshold: float = 1.5, sma_period: int = 50, dmi_threshold: float = 25) -> None:
        self._window = window
        self._threshold = threshold
        self._sma_period = sma_period
        self._dmi_threshold = dmi_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window * 2 + 100)

        if history.height < self._window * 2 + 100:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        symbols = [s for s in view.symbols if s in closes.to_series().unique()]
        symbols.sort()

        signals: dict[str, float] = {}
        for symbol in symbols[:5]:  # Limit to top 5 most actively traded stocks
            df = history.filter(pl.col("symbol") == symbol)

            if df.height < self._window * 2 + 100:
                continue

            close_series = df["adj_close"]
            volume_series = df["volume"]

            high_close = max(close_series.to_list()[-self._window:])
            low_close = min(close_series.to_list()[-self._window:])

            dmi_signal = False
            sma_signal = False

            if close_series[-1] > high_close or close_series[-1] < low_close:
                dmi_signal = True  # Check for breakout condition
                avg_volume = volume_series.mean().item()
                recent_volume = sum(volume_series.to_list()[-self._window:])
                sma_ratio = (recent_volume / avg_volume) >= self._threshold

                if sma_ratio and close_series[-1] > pl.col("adj_close").mean().over([pl.arange(0, self._sma_period)]).shift(-1)[-1]:
                    sma_signal = True  # Check for SMA condition

            if dmi_signal and sma_signal:
                signals[symbol] = 1.0 / len(symbols)  # Equal weight distribution

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest