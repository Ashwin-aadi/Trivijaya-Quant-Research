from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Seasonality effects can be exploited by identifying periods during the year "
        "when certain stocks tend to outperform. This strategy focuses on trading "
        "during these favorable times."
    )

    def __init__(self, window: int = 10, threshold: float = 0.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or len(history["symbol"].unique()) < 10:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        filtered_history = (
            history.filter(pl.col("symbol").is_in(symbols))
                  .sort("session_date")
                  .group_by("symbol")
                  .agg(
                      (pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1.0).alias("returns"),
                      (pl.col("session_date").dt.year()).alias("year")
                  )
        )

        yearwise_returns = (
            filtered_history.group_by("year")
                            .agg((pl.col("returns").mean()).alias("avg_return"))
        )

        positive_years = [y for y, avg in yearwise_returns.to_dicts() if avg > self._threshold]
        symbols_in_positive_years = [
            symbol
            for symbol, dates in history.group_by("symbol").collect()
                          .with_columns((pl.col("session_date").dt.year()).alias("year"))
                          .group_by("symbol", "year")
                          .agg(pl.col("returns").mean())
                          .filter(pl.col("avg_return").is_in(positive_years))
                          .select(["symbol"])
                          .to_dicts()
            for symbol in [s["symbol"] for s in dates]
        ]

        if not symbols_in_positive_years:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_in_positive_years)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_in_positive_years}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest