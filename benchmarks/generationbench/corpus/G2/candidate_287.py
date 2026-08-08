from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Dual momentum exploits the tendency for strong performers to remain strong and "
        "weak performers to continue weakening over time. By combining two momentum signals, "
        "we can identify stocks that are both historically strong and currently performing well."
    )

    def __init__(self, window1: int = 30, window2: int = 60) -> None:
        self._window1 = window1
        self._window2 = window2

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window1, self._window2))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window2)
        top_gainers = _top_performers(closes, window=self._window2)
        bottom_losers = _bottom_performers(closes, window=self._window2)

        momentum1 = history.select(
            pl.col("symbol").is_in(top_gainers),
            (pl.col("adj_close") / pl.col("adj_close").shift(self._window1) - 1.0).alias("r"),
        ).group_by("symbol").agg(pl.col("r").mean().alias("m1"))

        momentum2 = history.select(
            pl.col("symbol").is_in(bottom_losers),
            (pl.col("adj_close") / pl.col("adj_close").shift(self._window1) - 1.0).alias("r"),
        ).group_by("symbol").agg(pl.col("r").mean().alias("m2"))

        combined = (
            momentum1.join(
                momentum2,
                on="symbol",
                how="inner",
            )
            .with_columns(
                (pl.col("m1") + pl.col("m2")).alias("combined_momentum")
            )
            .sort("combined_momentum", descending=True)
            .head(self._window2 // 5)  # Select top 20% of combined momentum
        )

        picks = [row["symbol"] for row in combined.rows()]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _top_performers(closes: pl.DataFrame, window: int) -> list[str]:
    top_gainers = closes.select(
        pl.col("symbol"),
        (pl.col("adj_close") / pl.col("adj_close").shift(window) - 1.0).alias("return")
    ).group_by("symbol").agg(pl.col("return").mean().alias("avg_return"))

    return [row["symbol"] for row in top_gainers.sort("avg_return", descending=True).head(window // 5).rows()]


def _bottom_performers(closes: pl.DataFrame, window: int) -> list[str]:
    bottom_losers = closes.select(
        pl.col("symbol"),
        (pl.col("adj_close") / pl.col("adj_close").shift(window) - 1.0).alias("return")
    ).group_by("symbol").agg(pl.col("return").mean().alias("avg_return"))

    return [row["symbol"] for row in bottom_losers.sort("avg_return", descending=False).head(window // 5).rows()]