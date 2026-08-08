from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum20d(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by selecting stocks with recent "
        "positive price performance. It aims to capture short-term outperformance and benefit "
        "from the tendency of high-performing stocks to continue outperforming over a 1-3 month period."
    )

    def __init__(self, top_n: int = 20, window: int = 20) -> None:
        self._top_n = top_n
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_stocks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            last_close = float(values[-1])
            twenty_day_ago_close = float(values[0])
            return_percentage = ((last_close - twenty_day_ago_close) / twenty_day_ago_close) * 100
            top_stocks.append((symbol, return_percentage))

        top_stocks.sort(key=lambda x: x[1], reverse=True)
        top_stocks = [s for s, _ in top_stocks[: self._top_n]]
        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest