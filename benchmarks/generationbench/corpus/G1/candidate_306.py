from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Investing in stocks that have outperformed the broad market can provide "
        "excess returns. This strategy identifies symbols with the highest return against "
        "the NIFTY 100 index over a specified lookback period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        nifty_close = float(closes.select(pl.col("^NIFTY").last().item()))
        symbols = [s for s in view.symbols if s not in ["^NIFTY"]]

        symbol_returns: dict[str, float] = {}
        for symbol in symbols:
            symbol_closes = history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()
            if len(symbol_closes) < self._window:
                continue
            last_close = symbol_closes[-1]
            first_close = symbol_closes[0]
            return_ = (last_close - first_close) / first_close
            symbol_returns[symbol] = return_

        sorted_symbols = sorted(symbol_returns, key=symbol_returns.get, reverse=True)
        top_n_symbols = sorted_symbols[:5]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={(s + ".NS"): weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest