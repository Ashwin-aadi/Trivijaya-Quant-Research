from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength compared to the broader market "
        "can provide an edge in equity selection. Strong relative performance suggests favorable "
        "fundamentals or positive sentiment."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_close = view.latest_close()["NIFTYBANK"]
        relative_strengths: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol == "NIFTYBANK":
                continue
            latest_close = float(view.latest_close()[symbol])
            close_series = closes[symbol].drop_nulls().to_list()
            if len(close_series) < self._window:
                continue

            market_series = [float(v) for v in closes["NIFTYBANK"].drop_nulls().to_list()]
            relative_strength = (latest_close - min(market_close, latest_close)) / (
                max(market_close, latest_close) - min(market_close, latest_close)
            )
            if not (0 <= relative_strength <= 1):
                raise ValueError("Relative strength out of bounds")
            relative_strengths.append((symbol, relative_strength))

        sorted_rs = sorted(relative_strengths, key=lambda x: x[1], reverse=True)
        top_n_symbols = [rs[0] for rs in sorted_rs[:5]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest