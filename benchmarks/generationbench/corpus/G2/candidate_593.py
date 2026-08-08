from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "A breakout from a previous high or low can indicate the start of a new trend. "
        "If the market continues in that direction after a breakout, it may offer "
        "trading opportunities. This strategy identifies securities that have recently "
        "broken out and are likely to continue in the breakout direction."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.unique():
                continue
            prices = [float(v) for v in history.select("close", pl.col("symbol") == symbol)[
                      "close"].to_list()]
            if len(prices) < self._window + 2:
                continue

            breakout_high = max(prices[:-1])
            breakout_low = min(prices[:-1])
            latest_close = prices[-1]

            if (latest_close > breakout_high and latest_close >= history.select(
                    pl.col("close")).filter(pl.col("symbol") == symbol).sort(
                        "session_date", descending=True)[0, 1]) or (
                    latest_close < breakout_low and latest_close <= history.select(
                        pl.col("close")).filter(pl.col("symbol") == symbol).sort(
                            "session_date")[0, 1]):
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:self._top_n]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest