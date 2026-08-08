from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength compared to their peers within the NIFTY 100 "
        "universe are expected to outperform over time. This strategy selects stocks that have "
        "outperformed in the recent past relative to other stocks in the index."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            closes.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        top_n_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in returns.columns:
                continue
            values = [float(v) for v in returns[symbol].to_list()]
            if len(values) < self._window:
                continue
            mean_return = sum(values[-10:]) / 10.0  # Simple moving average of last 10 days

            # Compare the mean return with the overall mean return across all symbols to get relative strength
            if mean_return > pl.select(pl.col("r").mean()).item():
                top_n_symbols.append(symbol)

        top_n_symbols = top_n_symbols[:5]  # Select top 5 based on relative strength
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest