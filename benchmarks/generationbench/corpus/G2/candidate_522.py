from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "Companies with higher returns compared to their peers over a fixed period are "
        "expected to outperform. This is based on the idea that market inefficiencies allow "
        "dominant companies to maintain higher relative strength."
    )

    def __init__(self, lookback_period: int = 60) -> None:
        self._lookback_period = lookback_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._lookback_period)

        # Calculate returns for each symbol
        closes_with_returns = (
            closes
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._lookback_period) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .drop_nulls()
        )

        # Calculate mean return of the universe
        mean_return = closes_with_returns["return"].mean()

        # Rank symbols by their returns relative to the mean
        ranked_symbols = (
            closes_with_returns
            .select(
                pl.col("symbol"),
                (pl.col("return") - mean_return).abs().rank(method="dense", descending=True).alias("rank")
            )
        )

        top_n_symbols = [row["symbol"] for row in ranked_symbols.to_dict(orient="records") if row["rank"] <= 5]
        
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest