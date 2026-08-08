from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentumRSI(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: short-term momentum "
        "and relative strength index (RSI). By buying stocks with strong recent performance "
        "and high RSI values, we aim to capture both positive price trends and overbought conditions."
    )

    def __init__(self, lookback_short: int = 10, lookback_long: int = 50) -> None:
        self._lookback_short = lookback_short
        self._lookback_long = lookback_long

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_long + self._lookback_short - 1)

        if history.height < self._lookback_long + self._lookback_short - 1:
            return Signal(information_available_at=stamp, weights={})

        shorts = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("close") / pl.col("close").shift(self._lookback_short) - 1).alias("short_momentum"),
        )
        longs = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("close") / pl.col("close").shift(self._lookback_long) - 1).alias("long_momentum"),
        )

        rsi = _calculate_rsi(history, self._lookback_short)

        shorts = shorts.join(rsi, on="symbol", how="inner")
        shorts = shorts.sort(
            "short_momentum", descending=True
        ).sort("long_momentum", descending=True).select(pl.col("symbol"))

        if shorts.height < 5:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [s[0] for s in shorts.head(5).to_numpy()]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_rsi(history: pl.DataFrame, window: int) -> pl.LazyFrame:
    history = history.lazy().group_by("symbol").agg(
        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1).alias("returns")
    ).collect()
    rsi = history.select(
        pl.col("symbol"),
        ((pl.col("returns").rank(method="ordinal", descending=True, ascending=False) / window * 100)).alias("rsi"),
    )
    return rsi