from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks based on their relative strength against the broad market universe "
        "over a 3-month lookback period. It aims to capitalize on the persistence in stock performance and "
        "generate alpha by investing in outperforming stocks."
    )

    def __init__(self, window: int = 90, top_n_percent: float = 0.2) -> None:
        self._window = window
        self._top_n_percent = top_n_percent

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns for each stock
        history = (
            history
            .with_columns(
                (pl.col("close") - pl.col("open")) / pl.col("open").shift(-1).alias("return")
            )
            .sort("session_date", descending=False)
            .select(pl.col("symbol"), "return")
        )

        # Group by symbol and sum returns to get overall return for the period
        returns = history.groupby("symbol").agg(
            (pl.col("return").sum()).alias("total_return")
        ).sort("total_return", descending=True)

        top_n = int(len(view.symbols) * self._top_n_percent)
        top_symbols = [row[0] for row in returns.head(top_n).iter_rows()]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest