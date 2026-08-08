from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that financial markets tend to return to an average price over "
        "time. By identifying stocks that have deviated significantly from their historical mean, "
        "we can exploit the tendency for these prices to revert."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().mean().item()
        std_dev = closes.std().std().item()

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()][-self._window:]
            z_score = (recent_closes[-1] - mean_close) / std_dev
            if abs(z_score) > 2.0:  # Threshold for mean reversion opportunity
                signals[symbol] = -z_score * 0.5  # Allocate a portion of the portfolio

        weights = {s: w for s, w in signals.items() if w != 0}
        return Signal(
            information_available_at=stamp,
            weights={**weights},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest