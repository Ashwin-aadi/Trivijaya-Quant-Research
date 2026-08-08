from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Stocks with increased daily High-Low (H-L) range over 30 days often signal "
        "overbought or oversold conditions. Entering positions when the H-L range "
        "increases and exiting when it compresses, combined with a stop-loss mechanism, "
        "can capture profitable trades while managing risk."
    )

    def __init__(self, window: int = 30, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        h_l_ranges = []
        for symbol in view.symbols:
            history_symbol = history.select(
                pl.col("session_date"), pl.col(symbol).alias("close")
            )
            highs = history_symbol.with_column(history_symbol["close"].shift(-1).alias("high"))
            lows = history_symbol.with_column(history_symbol["close"].shift(-2).alias("low"))

            range_values = (
                (highs.select(pl.col("high") - pl.col("low")).to_series().to_list())
                + [highs.select(pl.col("high") - pl.col("close")).to_series().item()]
            )

            if len(range_values) < self._window:
                continue

            h_l_ranges.append((symbol, max(range_values)))

        sorted_h_l = sorted(h_l_ranges, key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in sorted_h_l[: self._top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        sma30 = view.closes().mean()
        close_prices = {symbol: float(view.latest_close()[symbol]) for symbol in view.symbols}

        valid_picks = []
        for pick in picks:
            if close_prices[pick] < sma30[pick]:
                valid_picks.append(pick)

        weight = 1.0 / len(valid_picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in valid_picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest