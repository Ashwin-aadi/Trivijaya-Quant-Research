from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets that have outperformed a broad market index in the recent past are more likely "
        "to continue outperforming due to momentum effects. By investing in the top performers, "
        "we can capture these returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns for each stock
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .head(self._window)
        )

        # Calculate the average return of the NIFTY 100 index during this period
        nifty_returns = history.select(
            pl.col("return").mean().alias("nifty_return")
        ).to_numpy()[0][0]

        # Rank symbols by their returns relative to the NIFTY 100 index
        rank_df = (
            history.groupby("symbol")
            .agg(
                (pl.col("return") - nifty_returns).rank(method="dense", descending=True)
                .alias("relative_strength")
            )
            .sort("relative_strength")
        )

        if rank_df.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in rank_df.to_dicts()[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest