from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for assets that have performed well "
        "relative to their peers in recent history to continue outperforming them in the future. "
        "This strategy aims to identify and invest in the top-performing stocks within a given period."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {symbol: 0.0 for symbol in view.symbols}
        for i in range(self._window - 1):
            date_i = stamp - date(2023, 1, 1) + date.timedelta(days=i)
            date_i_plus_1 = stamp - date(2023, 1, 1) + date.timedelta(days=i + 1)

            current_close = view.closes().filter(pl.col("session_date") == date_i).to_dict(as_series=False)
            previous_close = view.closes().filter(pl.col("session_date") == date_i_plus_1).to_dict(as_series=False)

            for symbol in view.symbols:
                if symbol not in current_close or symbol not in previous_close:
                    continue
                momentum_scores[symbol] += (float(current_close[symbol]) - float(previous_close[symbol])) / float(previous_close[symbol])

        top_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[: self._top_n]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest