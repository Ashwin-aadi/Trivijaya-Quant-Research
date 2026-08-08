from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression signals periods where price action has been restricted to a narrow "
        "range, which can precede breakouts or reversals. High dispersion within such ranges can "
        "indicate underlying market uncertainty and potential for greater movement in the future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_ratio: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            high_low_diff = max(values) - min(values)
            open_close_diff = abs(float(closes[symbol][-1]) - float(closes[symbol][0]))
            range_ratio.append(high_low_diff / (open_close_diff + 1e-8))

        mean_range_ratio = sum(range_ratio) / len(range_ratio)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            ratio = (max(values) - min(values)) / (open_close_diff + 1e-8)
            if ratio > mean_range_ratio * 2:
                picks.append(symbol)

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