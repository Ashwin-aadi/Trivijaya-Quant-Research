from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with strong relative performance against the broader market tend to continue "
        "performing well. This strategy identifies and invests in such stocks based on their "
        "recent strength relative to the NIFTY 100 index."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_close = float(view.latest_close()["^NSEI"])
        symbols = [symbol for symbol in view.symbols if symbol != "^NSEI"]
        relative_strengths: list[float] = []
        for symbol in symbols:
            closes = history.select(pl.col("adj_close")).to_numpy().flatten()
            if any(v is None for v in closes):
                continue
            nifty_closes = [float(c) for c in view.closes(lookback=self._window)[symbol].to_list()]
            if nifty_closes[-1] == 0 or nifty_close == 0:
                continue
            relative_strength = (closes[-1] - closes[0]) / (nifty_closes[-1] - nifty_closes[0])
            relative_strengths.append(relative_strength)

        top_n_symbols = [symbols[i] for i, _ in enumerate(sorted(zip(symbols, relative_strengths), key=lambda x: x[1], reverse=True))[:5]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest