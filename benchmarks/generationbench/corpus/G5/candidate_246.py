from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "Identifying stocks with strong relative performance against the broad market can "
        "lead to outperformance. This strategy selects the top-performing stocks based on their "
        "10-day cumulative return compared to the NIFTY 100 index."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(history["symbol"]) < 2:
            return Signal(information_available_at=stamp, weights={})

        nifty_closes = view.closes().select("NIFTYBANK_CLOSE")
        nifty_returns = (
            (nifty_closes.with_columns(
                (pl.col("NIFTYBANK_CLOSE") / pl.col("NIFTYBANK_CLOSE").shift(self._window) - 1.0).alias("r")
            ).sort("session_date", descending=False).select(pl.col("r")).to_list())
        )

        stock_returns = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            close_values = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].drop_nulls().to_list()]
            nifty_value = nifty_closes.filter(pl.col("NIFTYBANK_CLOSE").is_not_null()).select(pl.col("NIFTYBANK_CLOSE")[0]).item()
            if close_values[-1] / nifty_value - 1.0 > max(nifty_returns):
                stock_returns[symbol] = sum((close_values[i] / (close_values[max(0, i - self._window)] or 1) - 1.0 for i in range(self._window)))

        top_n_symbols = sorted(stock_returns.items(), key=lambda x: x[1], reverse=True)[:5]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s, _ in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest