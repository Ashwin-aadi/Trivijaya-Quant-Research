from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines 20-day mean reversion with a recent volatility factor to identify "
        "stocks that are both likely to revert and exhibit high recent volatility. The combination "
        "aims to capture opportunities in volatile markets."
    )

    def __init__(self, window_mean_reversion: int = 20, top_n: int = 5) -> None:
        self._window_mean_reversion = window_mean_reversion
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_mean_reversion)

        if history.height < self._window_mean_reversion:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores = {}
        volatility_scores = {}

        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue

            closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(closes) < self._window_mean_reversion:
                continue

            mean_reversion = (closes[-1] - sum(closes) / len(closes)) / abs(sum(closes) / len(closes))
            mean_reversion_scores[symbol] = mean_reversion

            volatility = pl.col("adj_close").std().alias("v")
            volatility_scores[symbol] = float(volatility)

        sorted_mean_reversion = sorted(mean_reversion_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_volatility = sorted(volatility_scores.items(), key=lambda x: abs(x[1]), reverse=True)

        selected_symbols = set()
        for symbol, mean_reversion in sorted_mean_reversion:
            if len(selected_symbols) >= self._top_n:
                break
            if volatility_scores[symbol] < 0.5:
                continue
            selected_symbols.add(symbol)

        weight = 1.0 / max(1, len(selected_symbols))
        return Signal(information_available_at=stamp, weights={s: weight for s in selected_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest