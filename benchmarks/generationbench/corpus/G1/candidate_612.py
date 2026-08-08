from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum involves investing in stocks that have performed well "
        "relative to the market over a certain period. This strategy leverages the tendency "
        "of outperforming stocks to continue outperforming in the short term."
    )

    def __init__(self, window: int = 20, lookback: int = 60) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window or history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        relative_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in history["symbol"].to_list():
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            recent_highs = history.filter(pl.col("symbol") == symbol)[
                f"high_{self._lookback}d"
            ].to_list()[0]
            recent_lows = history.filter(pl.col("symbol") == symbol)[
                f"low_{self._lookback}d"
            ].to_list()[0]

            strength = sum(
                [recent_closes[i] - min(recent_lows, key=lambda x: abs(x - recent_highs[i]))
                 for i in range(len(recent_highs))]
            ) / len(recent_highs)
            relative_strengths[symbol] = strength

        sorted_symbols = [
            s[0]
            for s in sorted(relative_strengths.items(), key=lambda item: item[1], reverse=True)
        ][: self._window]

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest