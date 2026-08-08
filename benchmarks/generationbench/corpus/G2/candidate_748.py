from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts to the mean over time. A stock that has deviated significantly from "
        "its historical price range is likely to return toward its average. This strategy "
        "identifies such deviations and takes positions accordingly."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("avg"))
            .select(["symbol", "avg"])
        )

        latest_closes = view.closes(lookback=self._window).with_columns(
            pl.col("session_date").alias("latest_session")
        )
        avg_latest = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0)
                .mean()
                .alias("deviation"),
            )
            .join(avg_close, on="symbol", how="left")
        )

        merged = avg_latest.join(latest_closes, on="symbol", how="inner")

        def score(row: pl.Series) -> float:
            deviation = row["deviation"]
            if pd.isna(deviation):
                return 0.0
            avg_close = row["avg"]
            latest_close = row["adj_close"]
            return (latest_close - avg_close) / deviation

        scores = merged.apply(score).to_series().sort(descending=True)

        top_symbols = [symbol for symbol, _ in scores.to_list()][:5]
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