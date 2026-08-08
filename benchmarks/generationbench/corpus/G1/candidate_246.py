from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Investing in securities that have outperformed the broad market can provide "
        "excess returns. This strategy selects the top-performing stocks relative to the index based on "
        "their 50-day cumulative return."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        index_close = float(history.filter(pl.col("symbol") == "NIFTY100").select("close").item())
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_performance: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            cumulative_return = (values[-1] / values[0]) - 1.0
            performance_ratio = cumulative_return / index_close
            symbol_performance[symbol] = performance_ratio

        top_performers: list[str] = sorted(symbol_performance, key=symbol_performance.get, reverse=True)[:5]
        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_performers}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest