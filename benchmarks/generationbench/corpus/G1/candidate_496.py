from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion occurs when prices return to the mean after deviating from it. "
        "This strategy identifies symbols that have moved away from their recent price range "
        "and bets on a return towards the mean."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            pl.col("adj_close").mean().alias("mean")
        )
        price_deviation = (
            history.lazy()
            .join(mean_close, on="symbol", how="left")
            .with_columns(
                (pl.col("adj_close") - pl.col("mean")).abs().alias("deviation")
            )
            .collect()
        )

        top_n_symbols = _select_top_deviators(price_deviation, n=self._window)
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


def _select_top_deviators(df: pl.DataFrame, n: int) -> list[str]:
    top_symbols = (
        df.sort("deviation", descending=True).head(n)["symbol"].to_list()
    )
    return top_symbols