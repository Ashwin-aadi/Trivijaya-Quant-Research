from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the recent strength of "
        "a stock's price and its volume trend. A combination of these signals aims to identify "
        "stocks with both a strong price performance and consistent volume."
    )

    def __init__(self, window_price: int = 20, threshold_price: float = 1.05) -> None:
        self._window_price = window_price
        self._threshold_price = threshold_price

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_price)
        if closes.height < self._window_price:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_price:
                continue

            # Calculate price strength ratio
            close_last = float(values[-1])
            close_mean = sum(values) / len(values)
            price_strength_ratio = close_last / close_mean

            # Check for high volume trend
            history = view.history(lookback=self._window_price).filter(
                (pl.col("symbol") == symbol)
            )
            volumes = [float(v) for v in history["volume"].to_list()]
            if len(volumes) < self._window_price:
                continue

            # Check for increasing volume trend
            last_volume = float(volumes[-1])
            mean_volume = sum(volumes) / len(volumes)
            is_volume_increasing = last_volume > (mean_volume * 1.05)

            if price_strength_ratio >= self._threshold_price and is_volume_increasing:
                picks.append(symbol)

        picks = list(set(picks))[:5]  # Remove duplicates and limit to top 5
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest