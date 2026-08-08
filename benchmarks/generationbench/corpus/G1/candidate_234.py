from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength against the broader market "
        "is expected to outperform over time. This strategy buys the top-performing stocks "
        "relative to the index."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate relative strength
        symbols = view.symbols
        rel_strength: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in closes.columns or "NIFTY" not in history.symbol.to_list():
                continue

            symbol_closes = [float(v) for v in closes[symbol].to_list()]
            nifty_closes = [
                float(v) for v in closes["NIFTY"].filter(pl.col("symbol") == "NIFTY").to_list()[0]
            ]
            if len(symbol_closes) < self._window or len(nifty_closes) < self._window:
                continue

            symbol_returns = [r for r in (symbol_closes[i] / symbol_closes[i - 1] - 1.0 for i in range(1, self._window))]
            nifty_returns = [r for r in (nifty_closes[i] / nifty_closes[i - 1] - 1.0 for i in range(1, self._window))]

            rel_strength[symbol] = sum(symbol_returns) / sum(nifty_returns)

        # Select top N symbols based on relative strength
        top_symbols = sorted(rel_strength.items(), key=lambda x: x[1], reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in [symbol for symbol, _ in top_symbols]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest