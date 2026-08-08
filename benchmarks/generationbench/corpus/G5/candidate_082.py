from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Historical data in the Indian market often exhibits seasonality effects, where "
        "certain times of the year consistently show favorable price movements. This strategy "
        "aims to capitalize on these seasonal patterns by identifying and trading symbols "
        "during their historically strong periods."
    )

    def __init__(self, window: int = 365, q2_min: float = 0.1, q3_min: float = 0.1) -> None:
        self._window = window
        self._q2_min = q2_min
        self._q3_min = q3_min

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        closes = history.select(pl.col("symbol").is_in(symbols).alias("in_symbols"))
        filtered_history = closes.filter(pl.col("in_symbols"))

        seasonality = (
            filtered_history.groupby("symbol")
                             .agg(
                                 (pl.col("close").rolling_mean(window_size=self._window / 4)).alias("mean_q1"),
                                 (pl.col("close").rolling_mean(window_size=self._window / 4, offset=30).alias("mean_q2")),
                                 (pl.col("close").rolling_mean(window_size=self._window / 4, offset=60).alias("mean_q3")),
                                 (pl.col("close").rolling_mean(window_size=self._window / 4, offset=90).alias("mean_q4"))
                             )
        )

        # Identify symbols with strong performance in Q2 and Q3
        q2_q3_strong = seasonality.filter(
            (pl.col("mean_q2") > self._q2_min * pl.col("mean_q1")) & 
            (pl.col("mean_q3") > self._q3_min * pl.col("mean_q2"))
        ).select(pl.col("symbol"))

        picks: list[str] = [row["symbol"] for row in q2_q3_strong.to_dicts()]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest