from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Trailing reversion exploits the tendency of stock prices to revert to their recent "
        "levels after deviating from them. If a stock has significantly deviated from its "
        "average price over the last 30 days, it is likely to move back towards that average."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the trailing average price
        symbols = [s for s in view.symbols if s in history.symbol.to_list()]
        avg_prices = (
            history.filter(pl.col("symbol").is_in(symbols))
                   .group_by("symbol")
                   .agg((pl.col("adj_close").mean()).alias("trailing_avg"))
                   .with_columns(
                       (pl.col("adj_close") / pl.col("trailing_avg") - 1.0).alias("deviation"),
                       ((pl.col("adj_close") - pl.col("trailing_avg")) / pl.col("trailing_avg") * 100.0).alias("percent_deviation")
                   )
        )

        # Identify symbols with significant deviation
        threshold = 1.5
        reversion_candidates = (
            avg_prices.filter(pl.col("percent_deviation").abs() > threshold)
                      .select("symbol", "deviation")
        ).to_dict(as_pandas=False)

        if not reversion_candidates:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to the candidates
        weight = 1.0 / len(reversion_candidates["symbol"])
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in reversion_candidates["symbol"].to_list()
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest