from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion30d(Strategy):
    rationale = (
        "This strategy seeks to exploit price reversion by identifying stocks that have "
        "deviated significantly from their 30-day average closing prices and are now "
        "potentially due for a bounce back towards the mean."
    )

    def __init__(self, window: int = 30, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        avg_closes = (
            history.group_by("symbol")
                   .agg(pl.col("adj_close").mean().alias("avg"))
        )
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in avg_closes.columns or symbol not in closes.columns:
                continue
            avg_close = float(avg_closes.filter(pl.col("symbol") == symbol)["avg"].item())
            daily_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(daily_closes) < self._window:
                continue
            latest_close = daily_closes[-1]
            if abs(latest_close - avg_close) / avg_close > 0.15:  # 15% deviation threshold
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest