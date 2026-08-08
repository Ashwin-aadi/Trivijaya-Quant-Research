from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency of stocks that have outperformed "
        "in the recent past to continue to outperform. This strategy ranks assets based on "
        "their returns over a short period and allocates capital to the top performers."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns for each symbol
        history_with_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .select(["symbol", "return"])
        )

        # Rank the symbols by return in descending order
        ranked = history_with_returns.with_columns(
            pl.col("return").rank(method="ordinal", descending=True).alias("rank")
        ).filter(pl.col("rank") <= self._window)

        top_n_symbols = [row["symbol"] for row in ranked.to_dicts()]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest