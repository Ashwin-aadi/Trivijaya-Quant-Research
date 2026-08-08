from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks based on their relative strength against the broader market can "
        "help in identifying outperformers. This strategy focuses on selecting symbols that have "
        "outperformed the average performance of the NIFTY 100 constituents."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_mean = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).mean()
        symbols_history = history.select(
            pl.col("symbol").alias("Symbol"), "adj_close"
        ).pivot(index="session_date", columns="Symbol", values="adj_close")

        if symbols_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_returns = (symbols_history.to_numpy() / symbols_history.shift(1).to_numpy() - 1.0)
        outperformers: list[str] = []
        for _, row in pl.DataFrame(symbol_returns).rows():
            if any(ret > nifty_mean for ret in row):
                outperformers.extend([symb for symb, ret in zip(symbols_history.columns, row) if ret > nifty_mean])

        unique_outperformers = set(outperformers)
        weight = 1.0 / len(unique_outperformers)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in unique_outperformers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest