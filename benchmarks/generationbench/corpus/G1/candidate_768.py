from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy identifies the top-performing stocks based on historical returns. "
        "Stocks with higher momentum are expected to continue outperforming in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width == 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            closes.drop_nulls()
            .select(
                [pl.col(symbol).shift(-1) / pl.col(symbol) - 1.0 for symbol in view.symbols]
            )
            .transpose()  # Transpose to get a wide format with symbols as columns and returns as rows
            .with_columns(pl.Series("session_date", closes["session_date"].to_list()))
        )

        # Select the most recent return values
        latest_returns = (
            returns.sort("session_date", descending=True).select(
                [pl.col(symbol) for symbol in view.symbols]
            ).head(1)
        )

        top_n_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in latest_returns.columns:
                continue
            if float(latest_returns[symbol].to_list()[0]) > 0.02:  # Consider top 2% performers
                top_n_symbols.append(symbol)

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