from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a strong breakout, the stock often consolidates and then continues in "
        "the direction of the initial movement. This strategy identifies such setups "
        "and allocates capital to those stocks that show signs of continued momentum."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_dates = _find_breakout_dates(closes)
        continuation_symbols = []
        for symbol in view.symbols:
            if symbol not in breakout_dates.columns:
                continue
            if len(breakout_dates[symbol].drop_nulls().to_list()) < self._window + 1:
                continue

            breakout_date = breakout_dates[symbol][0]
            last_close = float(closes[symbol][-1])
            first_close_after_breakout = float(closes[symbol][breakout_date:].shift(1).head(1)[0])

            if (last_close - first_close_after_breakout) / first_close_after_breakout > self._threshold:
                continuation_symbols.append(symbol)

        continuation_symbols = continuation_symbols[:5]
        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in continuation_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _find_breakout_dates(closes: pl.DataFrame) -> pl.DataFrame:
    breakout_dates: dict[str, int] = {}
    for symbol in closes.columns[1:]:
        values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
        if len(values) < 2 * (len(values) // 2 + 1):
            continue
        max_value = max(values[: len(values) // 2])
        breakout_date = next(i for i, value in enumerate(values[len(values) // 2:]) if value > max_value)
        breakout_dates[symbol] = len(values) // 2 + breakout_date

    return pl.DataFrame({"symbol": list(breakout_dates.keys()), "breakout_date": list(breakout_dates.values())})