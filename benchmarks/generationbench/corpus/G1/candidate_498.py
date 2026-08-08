from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength against the NIFTY 100 index "
        "can help in identifying outperformers. The strategy leverages historical price data "
        "to rank assets and select top performers."
    )

    def __init__(self, lookback_days: int = 20) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_history = history.filter(pl.col("symbol") == "NIFTY 100")
        if nifty_history.height < self._lookback_days:
            return Signal(information_available_at=stamp, weights={})

        close_prices = view.closes(lookback=self._lookback_days)
        if close_prices.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute the relative strength ratio
        relative_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol == "NIFTY 100":
                continue

            nifty_close = [float(v) for v in nifty_history["adj_close"].drop_nulls().to_list()]
            close_prices_col = [float(v) for v in close_prices[symbol].drop_nulls().to_list()]

            if len(nifty_close) != self._lookback_days or len(close_prices_col) != self._lookback_days:
                continue

            ratio = sum(close_prices_col) / sum(nifty_close)
            relative_strengths[symbol] = ratio

        # Rank symbols by their relative strength
        ranked_symbols = sorted(relative_strengths.items(), key=lambda x: x[1], reverse=True)

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_n = min(len(ranked_symbols), 5)  # Select the top N symbols
        weight = 1.0 / len(top_n)
        selected_symbols = [s for s, _ in ranked_symbols[:top_n]]

        return Signal(
            information_available_at=stamp, weights={symbol: weight for symbol in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest