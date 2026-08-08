from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price levels revert to their mean over time. If a stock has been significantly above "
        "its trailing average, it is likely to fall back towards that average, and vice versa. "
        "This strategy exploits such reversions by betting against the recent trend."
    )

    def __init__(self, window: int = 50, mean_window: int = 20) -> None:
        self._window = window
        self._mean_window = mean_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + self._mean_window - 1:
            return Signal(information_available_at=stamp, weights={})

        means = closes.lazy().group_by("symbol").agg(
            (pl.col("adj_close").shift(-self._mean_window).mean()).alias(f"mean_{self._mean_window}")
        ).collect()

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in means.columns or symbol not in closes.columns:
                continue
            mean_value = means[f"mean_{self._mean_window}"].to_list()[0]
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()[-self._window:]]
            if len(recent_closes) < self._window:
                continue
            price_trend = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            reversion_signal = (price_trend > 0.05 and mean_value > recent_closes[-1]) or \
                              (price_trend < -0.05 and mean_value < recent_closes[-1])
            if reversion_signal:
                signals[symbol] = 1.0 / len(signals)

        return Signal(information_available_at=stamp, weights={k: v for k, v in signals.items() if v > 0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest