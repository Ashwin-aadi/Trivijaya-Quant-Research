from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion occurs when prices or returns tend to move towards the mean. "
        "In a short horizon like 10 days, stocks that have deviated significantly from their "
        "mean price are likely to return to it, providing an opportunity for profit."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean_close"))
        )
        latest_closes = view.closes(lookback=self._window + 1)
        
        deviations = pl.DataFrame()
        for symbol in view.symbols:
            if symbol not in mean_close.columns or symbol not in latest_closes.columns:
                continue
            mean_val = float(mean_close[symbol]["mean_close"])
            latest_val = float(latest_closes[symbol].drop_nulls().to_list()[-1])
            deviations = deviations.with_column(
                pl.Series([symbol, abs(latest_val - mean_val)]).transpose().explode()
            )

        if deviations.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        sorted_deviations = (
            deviations.sort("field_1", descending=True)
                      .head(5)  # Select the top 5 symbols with highest deviation
        )
        
        weight = 1.0 / len(sorted_deviations["field_0"].to_list())
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_deviations["field_0"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).collect()[0][0]
    assert isinstance(newest, date)
    return newest