from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy invests in the top-performing stocks based on their recent price "
        "momentum. The idea is that stocks with strong past performance are likely to continue "
        "outperforming the market."
    )

    def __init__(self, window: int = 20, n_top_stocks: int = 5) -> None:
        self._window = window
        self._n_top_stocks = n_top_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )

        # Group by symbol and calculate the mean return over the window period
        grouped = history.group_by("symbol").agg(pl.col("return").mean().alias("avg_return"))
        if grouped.height < self._n_top_stocks + 1:
            return Signal(information_available_at=stamp, weights={})

        # Sort symbols by average return in descending order
        sorted_symbols = grouped.sort("avg_return", descending=True)

        top_n_symbols = [row[0] for row in sorted_symbols.head(self._n_top_stocks).to_dict(as_pandas=False)["symbol"]]
        
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest