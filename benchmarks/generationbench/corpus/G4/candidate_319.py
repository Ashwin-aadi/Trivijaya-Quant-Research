from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "This strategy targets sectors where stock prices exhibit high dispersion but are "
        "experiencing low range compression. High dispersion suggests potential inefficiencies, "
        "while narrowing ranges indicate a setup for breakout or consolidation opportunities."
    )

    def __init__(self, lookback_vol: int = 20, lookback_range: int = 30, top_n: int = 5) -> None:
        self._lookback_vol = lookback_vol
        self._lookback_range = lookback_range
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_range + 5)
        if history.height < self._lookback_range + 5:
            return Signal(information_available_at=stamp, weights={})

        # Calculate historical volatility
        volatility = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).alias("daily_return")
            )
            .with_columns(pl.col("daily_return").abs().mean().over("symbol").alias("avg_daily_return"))
            .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(self._lookback_vol) - 1.0).alias("volatility"))
            .sort("session_date", descending=False)
        )

        # Calculate price range
        high_low_range = (
            history.group_by("symbol")
                   .agg(
                       (pl.col("high") - pl.col("low")).max().alias("range_max"),
                       (pl.col("high").max() - pl.col("low").min()).alias("total_range")
                   )
        )

        # Combine and rank based on criteria
        combined = history.join(high_low_range, on="symbol", how="left")
        combined = (
            combined.join(volatility.select("symbol", "volatility"), on="symbol", how="left")
            .sort(["session_date", "volatility"], descending=[False, True])
            .group_by("symbol")
            .agg([
                (pl.col("range_max").shift(1) - pl.col("range_max")).abs().mean().alias("recent_range_change"),
                (pl.col("total_range") / self._lookback_range).mean().alias("avg_total_range")
            ])
            .with_columns((pl.col("volatility") > 0.5 * pl.col("volatility").shift(1)).alias("high_volatility"))
            .sort(["recent_range_change", "avg_total_range"], descending=[True, False])
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if combined.filter(pl.col("symbol") == symbol).height < 5 or combined.filter(pl.col("symbol") == symbol)["high_volatility"].sum() == 0:
                continue
            if (combined.filter(pl.col("symbol") == symbol)["recent_range_change"] <= 0) and (
                    combined.filter(pl.col("symbol") == symbol)["avg_total_range"] < combined.select("avg_total_range").mean()):
                picks.append(symbol)
            elif (combined.filter(pl.col("symbol") == symbol)["high_volatility"]) and (
                    combined.filter(pl.col("symbol") == symbol)["recent_range_change"] > 0):
                picks.append(symbol)

        picks = picks[: self._top_n]
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