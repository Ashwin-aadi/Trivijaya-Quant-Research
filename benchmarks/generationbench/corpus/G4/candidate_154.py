from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Seasonality in the Indian market can be harnessed by identifying stocks that exhibit "
        "historical positive returns at specific times of the year. This is based on the theory "
        "that certain dates or periods have a higher likelihood of favorable market conditions."
    )

    def __init__(self, lookback: int = 5 * 365, top_n: int = 20) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            data = history.filter(pl.col("symbol") == symbol).select(
                ["session_date", "adj_close"]
            )
            if data.height < self._lookback:
                continue

            # Compute technical indicators
            rsi = _compute_rsi(data, lookback=self._lookback // 2)
            sma_50 = (data["adj_close"].rolling_mean(window_size=50)).alias("sma_50")
            data = data.with_columns(sma_50)

            # Identify key dates
            earnings_dates = _find_key_dates(data, "earnings", stamp)
            festival_dates = _find_key_dates(data, "festival", stamp)

            # Rank stocks based on performance metrics
            scores = (
                data.join(
                    rsi,
                    on="session_date",
                    how="inner",
                )
                .join(
                    earnings_dates,
                    on="session_date",
                    how="inner",
                )
                .join(
                    festival_dates,
                    on="session_date",
                    how="inner",
                )
                .select(
                    pl.col("symbol"),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
                    (pl.col("sma_50") > pl.col("adj_close")).alias("above_sma"),
                    _ranked_by(
                        "return", descending=True
                    ).alias("return_rank"),
                    _ranked_by(
                        "above_sma", descending=False
                    ).alias("above_sma_rank"),
                )
            )

            # Calculate final score and select top N stocks
            scores = scores.with_columns(
                (0.6 * pl.col("return_rank") + 0.4 * pl.col("above_sma_rank")).alias("final_score")
            )
            selected = scores.sort(by="final_score", descending=True).head(self._top_n)

            picks.extend([row["symbol"] for _, row in selected.iter_rows()])

        picks = list(set(picks))[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
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


def _compute_rsi(data: pl.DataFrame, lookback: int) -> pl.LazyFrame:
    delta = data["adj_close"].diff().shift(-1)
    gain = (delta.where(delta > 0, other=0)).rolling_mean(window_size=lookback)
    loss = (-delta.where(delta < 0, other=0)).rolling_mean(window_size=lookback)

    rsi = ((100 - (100 / (1 + gain / loss))).alias("rsi"))
    return data.join(gain.select([pl.col("session_date"), gain]), on="session_date", how="inner").join(
        loss.select([pl.col("session_date"), loss]), on="session_date", how="inner"
    ).with_columns(rsi)


def _find_key_dates(data: pl.DataFrame, keyword: str, stamp: date) -> pl.LazyFrame:
    return (
        data.filter((pl.col("session_date") >= (stamp - pl.duration(days=365))) & (pl.col("session_date") < stamp))
        .group_by("symbol")
        .agg(
            ((pl.col("session_date") == keyword).sum().alias(f"{keyword}_count")),
        )
    ).filter(pl.col(f"{keyword}_count").gt(0)).select(["symbol"])


def _ranked_by(col: str, descending: bool) -> pl.Expr:
    return (pl.col(col) / pl.col(col).max()).rank(method="ordinal", descending=descending)