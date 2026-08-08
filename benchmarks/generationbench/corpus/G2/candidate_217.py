from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that after a security has been extremely volatile and moved "
        "far from its mean price over the recent past, it will tend to revert back towards this "
        "mean. In an efficient market with no predictability, such reversions should offer "
        "trading opportunities."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_close = float(sum(values) / self._window)
            z_score = (values[-1] - mean_close) / pl.col("adj_close").std()
            if abs(z_score) > self._threshold:
                picks.append(symbol)

        picks = list(set(picks))  # Remove duplicates
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