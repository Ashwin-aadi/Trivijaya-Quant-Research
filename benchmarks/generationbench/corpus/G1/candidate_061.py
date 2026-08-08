from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the tendency of stock prices to revert to their "
        "mean over a short period. This strategy buys underperforming stocks and sells outperforming ones."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().to_dict().get("adj_close", 0.0)

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            price_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(price_series) < self._window:
                continue

            mean_reversion_score = (price_series[-1] - mean_close) / mean_close
            if abs(mean_reversion_score) > 0.2:  # Threshold for significant reversion
                signals[symbol] = mean_reversion_score

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        adjusted_signals = {s: (weight * v) for s, v in signals.items()}
        return Signal(
            information_available_at=stamp, weights=adjusted_signals
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest