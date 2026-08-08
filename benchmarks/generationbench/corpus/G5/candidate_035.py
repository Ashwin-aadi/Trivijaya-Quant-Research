from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term and long-term momentum signals can help in identifying "
        "stocks that are trending in a consistent direction over both periods."
    )

    def __init__(self, short_window: int = 10, long_window: int = 50) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes_long = view.closes(lookback=self._long_window)
        closes_short = view.closes(lookback=self._short_window)

        if (closes_long.height < self._long_window or
                closes_short.height < self._short_window):
            return Signal(information_available_at=stamp, weights={})

        short_moments = _calculate_momentum(closes_short)
        long_moments = _calculate_momentum(closes_long)

        combined_scores = pl.DataFrame({
            "symbol": short_moments["symbol"],
            "score": (short_moments["momentum"] * 0.6) + (long_moments["momentum"] * 0.4)
        })

        top_symbols = combined_scores.sort("score", descending=True)["symbol"].to_list()[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_momentum(closes: pl.DataFrame) -> pl.DataFrame:
    symbols = closes.columns[1:]
    moments = []
    for symbol in symbols:
        prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
        if len(prices) < 2 or not all(prices):
            continue
        last_price = prices[-1]
        momentum = (last_price - prices[0]) / max(prices)
        moments.append({"symbol": symbol, "momentum": momentum})
    return pl.DataFrame(moments)