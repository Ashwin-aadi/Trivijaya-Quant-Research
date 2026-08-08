from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets that have outperformed the broader market in recent history are likely to "
        "continue to outperform due to momentum effects. This strategy allocates capital to "
        "the top performers in order to capture these positive returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(closes.columns) <= 1:
            return Signal(information_available_at=stamp, weights={})

        latest_close = {s: float(v) for s, v in zip(view.symbols, view.latest_close().values())}
        avg_market = sum(latest_close[s] for s in closes.columns if s != "NIFTY") / len(closes.columns) - 1.0

        relative_strength = {
            symbol: (latest_close[symbol] - latest_close["NIFTY"]) / latest_close["NIFTY"]
            for symbol in view.symbols
        }

        top_symbols = sorted(relative_strength, key=lambda s: relative_strength[s], reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest