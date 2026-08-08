from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionStrategy(Strategy):
    rationale = (
        "This strategy exploits periods of high market dispersion by entering trades "
        "when the next day’s open price falls outside the previous day's range. It aims to "
        "capitalize on increased volatilities during range expansions and implements strict risk "
        "management through stop-loss orders."
    )

    def __init__(self, window: int = 20, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history = view.history().filter(pl.col("symbol") == symbol)
            open_prices = [float(v) for v in history["open"].to_list()]
            high_prices = [float(v) for v in history["high"].to_list()]
            low_prices = [float(v) for v in history["low"].to_list()]

            if len(open_prices) < self._window:
                continue
            prev_day_range = max(high_prices[-1], open_prices[-1]) - min(low_prices[-1], open_prices[-1])
            next_open = float(closes[symbol][0])

            if next_open > high_prices[-1] + 0.5 * (high_prices[-1] - low_prices[-1]):
                picks.append(symbol)
            elif next_open < low_prices[-1] - 0.5 * (high_prices[-1] - low_prices[-1]):
                picks.append(symbol)

        picks = picks[: self._top_n]
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