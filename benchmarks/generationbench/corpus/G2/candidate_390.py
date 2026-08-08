from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed their peers over a recent period are more likely "
        "to continue this trend. This strategy selects the top performers based on "
        "their price appreciation relative to the market index."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns relative to the NIFTY 100 index
        nifty_closes = view.history().select(
            pl.col("symbol") == "NIFTY_100"
        ).get_column("adj_close").to_list()
        if len(nifty_closes) < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_returns = (nifty_closes[-1] / nifty_closes[0]) - 1.0

        symbol_returns = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            returns = (values[-1] / values[0]) - 1.0
            symbol_returns[symbol] = returns

        # Identify top performers based on their relative strength
        top_performers = sorted(symbol_returns.items(), key=lambda x: x[1], reverse=True)[:5]

        if not top_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in top_performers]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest