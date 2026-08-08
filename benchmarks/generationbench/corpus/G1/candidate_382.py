from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Stocks often exhibit predictable seasonal patterns due to various macroeconomic factors. "
        "By identifying these trends, we can exploit them for trading opportunities."
    )

    def __init__(self, window: int = 365, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            yearly_data = _chunk_into_yearly(values, self._window)
            mean_return = sum(yearly_data.values()) / len(view.symbols)
            trends[symbol] = max([max(chunk) - min(chunk) for chunk in yearly_data.keys()])

        sorted_trends = {k: v for k, v in sorted(trends.items(), key=lambda item: item[1], reverse=True)}
        picks = list(sorted_trends.keys())[: self._top_n]
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


def _chunk_into_yearly(data: list[float], window: int) -> dict[date, float]:
    chunks: dict[date, float] = {}
    start_date = view.as_of - pl.duration.Years(10)  # Assuming a large span to cover multiple years
    current_chunk_start = None

    for i, close in enumerate(data):
        session_date = view.history().filter(pl.col("adj_close") == close)["session_date"].item()
        if start_date + pl.duration.Days(window) < session_date:
            if current_chunk_start is not None:
                chunks[current_chunk_start] = (max(data[i - window:i]) - min(data[i - window:i]))
            current_chunk_start = session_date
    return chunks