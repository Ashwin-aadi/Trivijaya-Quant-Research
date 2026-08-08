from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "This strategy selects stocks based on their relative strength against the broader NIFTY 50 index. "
        "It aims to capitalize on momentum and value by entering positions in outperforming stocks and exiting "
        "when conditions suggest a reversal or risk of significant drawdown."
    )

    def __init__(self, window: int = 30, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or "NIFTY 50" not in history["symbol"].to_list():
            return Signal(information_available_at=stamp, weights={})

        nifty_50_history = history.filter(pl.col("symbol") == "NIFTY 50")
        stock_history = history.filter(pl.col("symbol").is_in(view.symbols))

        if nifty_50_history.height < self._window or stock_history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_50_closes = nifty_50_history["adj_close"].to_list()[-self._window:]
        stock_closes = [float(c) for symbol in view.symbols for c in
                        stock_history.filter(pl.col("symbol") == symbol)["adj_close"].drop_nulls().to_list()]

        if len(stock_closes) < self._window:
            return Signal(information_available_at=stamp, weights={})

        rsi_values = [(c / nifty_50_closes[i] - 1.0 for i, c in enumerate(stock_closes))]

        top_n_indices = sorted(range(len(rsi_values)), key=lambda k: rsi_values[k], reverse=True)[:self._top_n]
        picks = [view.symbols[i] for i in top_n_indices if stock_history.filter(pl.col("symbol") == view.symbols[i]).select(
            pl.col("adj_close").mean()).to_list()[-1] > 0.95 * nifty_50_closes[-1]]

        weight = 1.0 / len(picks) if picks else 0.0
        return Signal(information_available_at=stamp, weights={s: weight for s in picks})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest