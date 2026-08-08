from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines the recent strength of a stock with its relative strength "
        "against the NIFTY 100 index. A strong closing price and high relative strength can "
        "indicate future positive performance."
    )

    def __init__(self, window: int = 20, threshold: float = 5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_100_close = [float(v) for v in closes["^NSEI"].drop_nulls().to_list()]
        symbol_closes = {symbol: [float(v) for v in closes[symbol].drop_nulls().to_list()] for symbol in view.symbols}

        picks: list[str] = []
        for symbol, values in symbol_closes.items():
            if len(values) < self._window:
                continue
            recent_close = values[-1]
            nifty_100_recent_close = nifty_100_close[-1]
            if recent_close >= max(values):
                relative_strength = (recent_close / nifty_100_recent_close - 1.0) * 100
                if relative_strength > self._threshold:  # Consider stocks with strong relative performance
                    picks.append(symbol)

        picks = picks[:5]
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