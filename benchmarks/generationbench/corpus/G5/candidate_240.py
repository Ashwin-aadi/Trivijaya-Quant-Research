from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum involves buying the top performers and selling the "
        "underperformers from the previous period. This strategy captures trends in relative performance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        returns = history.select(
            pl.col("symbol"),
            (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
        )

        # Filter out symbols with no data for the full window
        returns = returns.filter(pl.col("return").is_not_null())
        symbols_with_full_data = set(returns["symbol"].to_list())

        top_symbols: list[str] = []
        bottom_symbols: list[str] = []

        for symbol in view.symbols:
            if symbol not in symbols_with_full_data or symbol not in returns.columns:
                continue

            return_val = float(returns.filter(pl.col("symbol") == symbol)["return"][0])
            if return_val > 0.0:
                top_symbols.append(symbol)
            else:
                bottom_symbols.append(symbol)

        # Select top and bottom performers
        top_n = min(self._window, len(top_symbols))
        bottom_n = min(self._window, len(bottom_symbols))

        top_weights = {s: 1.0 / top_n for s in top_symbols[:top_n]}
        bottom_weights = {s: -1.0 / bottom_n for s in bottom_symbols[:bottom_n]}

        final_weights = {**top_weights, **bottom_weights}
        return Signal(information_available_at=stamp, weights=final_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest