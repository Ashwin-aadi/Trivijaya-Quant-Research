from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength against the broad universe "
        "helps to identify outperformers. This strategy focuses on selecting the top N "
        "stocks based on their performance compared to the average performance of all "
        "stocks in the market."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        avg_close = float(closes.select(pl.mean("adj_close")).to_numpy()[0][0])
        relative_strengths: list[tuple[str, float]] = []

        for symbol in symbols:
            if symbol not in history["symbol"].to_list():
                continue
            latest_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            avg_latest_close = sum(latest_closes) / len(latest_closes)
            strength = (avg_latest_close - avg_close) / avg_close
            relative_strengths.append((symbol, strength))

        sorted_strengths = sorted(relative_strengths, key=lambda x: x[1], reverse=True)
        top_symbols = [x[0] for x in sorted_strengths[: self._top_n]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest