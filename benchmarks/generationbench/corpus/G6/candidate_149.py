from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedAmbitiousBreakoutContinuation(Strategy):
    rationale = (
        "This strategy integrates elements from both conservative and ambitious designs to "
        "create a robust yet manageable approach tailored for the Indian market. It uses breakouts"
        " identified by crossing above/below 20-day moving averages with significant volume, ensuring strong entry signals while maintaining simplicity."
    )

    def __init__(self, window: int = 20, top_n: int = 15) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.select("symbol").to_series().unique():
                continue
            adj_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].drop_nulls().to_list()]

            if len(adj_closes) < self._window + 1:
                continue

            ma_20 = sum(adj_closes[-self._window:]) / self._window
            latest_price = adj_closes[-1]
            high, low = max(adj_closes), min(adj_closes)
            volume = history.filter(pl.col("symbol") == symbol)["volume"].sum()

            if (latest_price > ma_20 and latest_price >= high) or \
               (latest_price < ma_20 and latest_price <= low):
                if latest_price * 0.95 <= history.filter(pl.col("symbol") == symbol)["adj_close"].sort(descending=True).head(1)[0] <= latest_price * 1.05:
                    continue
                elif volume > pl.col("volume").mean().sum():
                    picks.append(symbol)

        picks = picks[: self._top_n]
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