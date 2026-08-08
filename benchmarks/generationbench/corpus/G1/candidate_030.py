from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength against the broader market "
        "can provide excess returns. This strategy leverages the idea that high performers "
        "relative to the index tend to continue their upward trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the relative strength for each symbol
        rel_strengths = {}
        nse_index_close = float(closes.select(pl.col("NIFTY_50").max().alias("index_close"))[0]["index_close"])
        
        for symbol in view.symbols:
            if "NIFTY_50" != symbol and symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            
            # Calculate the relative strength ratio
            last_price = values[-1]
            average_price = sum(values[-self._window:]) / self._window
            rel_strength = (last_price - average_price) / average_price * nse_index_close / last_price
            rel_strengths[symbol] = rel_strength

        # Sort symbols by relative strength in descending order
        sorted_symbols = sorted(rel_strengths, key=rel_strengths.get, reverse=True)
        
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={(symbol): weight for symbol in sorted_symbols[:5]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest