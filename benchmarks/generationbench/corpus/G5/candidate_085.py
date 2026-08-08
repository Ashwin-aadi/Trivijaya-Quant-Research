from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion identifies assets that have deviated significantly "
        "from their historical average price over a 10-day window. These assets are then "
        "likely to revert to the mean in the near term, providing an opportunity for profit."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        means: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            mean_close = sum(values) / self._window
            means[symbol] = mean_close

        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in means or symbol not in closes.columns:
                continue
            latest_close = float(closes[symbol].to_list()[-1])
            mean_close = means[symbol]
            z_score = (latest_close - mean_close) / mean_close

            if abs(z_score) > 0.3:  # Threshold for deviation from mean
                signals[symbol] = z_score

        sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_signals][:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest