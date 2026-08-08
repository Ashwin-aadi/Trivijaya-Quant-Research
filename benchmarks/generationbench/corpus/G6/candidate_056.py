from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy identifies significant price movements confirmed by trading volumes "
        "exceeding historical norms. It aims to capitalize on strong market trends by entering "
        "positions when a stock's daily high or low breaches predefined thresholds with "
        "corresponding volume exceeding its moving average."
    )

    def __init__(self, window: int = 20, threshold_percentage: float = 3.0) -> None:
        self._window = window
        self._threshold_percentage = threshold_percentage / 100

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            hist = history.filter(pl.col("symbol") == symbol)
            open_price = float(hist.select(pl.col("open")[0]))
            close_price = float(hist.select(pl.col("close")[-1]))
            high_price = float(hist.select(pl.col("high").max()))
            low_price = float(hist.select(pl.col("low").min()))
            volume = float(hist.select(pl.col("volume")[-1]))

            prev_range = abs(high_price - low_price)
            threshold_high = open_price + (prev_range * self._threshold_percentage / 100)
            threshold_low = open_price - (prev_range * self._threshold_percentage / 100)

            if high_price >= threshold_high and volume > history.filter(pl.col("symbol") == symbol).select(pl.col("volume").mean()).item():
                signals[symbol] = 0.05
            elif low_price <= threshold_low and volume > history.filter(pl.col("symbol") == symbol).select(pl.col("volume").mean()).item():
                signals[symbol] = 0.05

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        for symbol in signals.keys():
            signals[symbol] /= total_weight
        return Signal(
            information_available_at=stamp,
            weights=signals,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, pl.Datetime)
    return newest.date()