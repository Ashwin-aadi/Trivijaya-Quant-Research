from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression can indicate increased market volatility and potential for breakout "
        "in either direction. High dispersion in recent prices suggests that the current price is "
        "likely near a support or resistance level, which could lead to a strong move."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        symbols = closes.columns

        dispersion_scores: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            high_low_diff = max(prices) - min(prices)
            recent_high_low_diff = (max(prices[-self._window:]) - min(prices[-self._window:])) / 2.0
            if recent_high_low_diff == 0:
                continue
            dispersion_score = high_low_diff / recent_high_low_diff
            dispersion_scores[symbol] = dispersion_score

        sorted_scores = sorted(dispersion_scores.items(), key=lambda item: -item[1])
        top_symbols = [s for s, _ in sorted_scores[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest