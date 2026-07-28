"""Smoothed trend-following strategy for NIFTY 100 constituents."""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class SmoothedTrendFollow(Strategy):
    """Trades in the direction of a smoothed trend line built from the price history.

    Daily closes are noisy, so a smoothed trend line filters out single-session swings: a name
    trading above its smoothed line is judged to be in an uptrend and a name trading below it is
    judged to be in a downtrend. Positions are taken with the trend, in the names showing the
    clearest separation from their own smoothed line.
    """

    rationale = (
        "Daily closing prices contain enough single-session noise that a raw price comparison "
        "is an unreliable trend signal; smoothing the series removes the noise and leaves the "
        "underlying direction. Stocks trading clearly above their smoothed trend line are in a "
        "sustained uptrend and are bought; those clearly below are avoided."
    )

    def __init__(self, panel: pl.DataFrame, window: int = 11, top_n: int = 10) -> None:
        self.top_n = top_n
        # Trend line smoothed once from the full historical panel and cached per session so
        # the same reference line can be looked up on every trading day.
        smoothed = panel.sort(["symbol", "session_date"]).with_columns(
            pl.col("adj_close")
            .rolling_mean(window_size=window, center=True)
            .over("symbol")
            .alias("trend_line")
        )
        self._trend: dict[tuple[str, date], float] = {
            (row["symbol"], row["session_date"]): row["trend_line"]
            for row in smoothed.select("symbol", "session_date", "trend_line").iter_rows(
                named=True
            )
            if row["trend_line"] is not None
        }

    def generate(self, view: MarketView) -> Signal:
        history = view.history()
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        last_date = history["session_date"].max()
        gaps: dict[str, float] = {}
        for symbol, price in view.latest_close().items():
            trend_line = self._trend.get((symbol, last_date))
            if trend_line:
                gaps[symbol] = (price - trend_line) / trend_line
        ranked = sorted(gaps.items(), key=lambda kv: kv[1], reverse=True)[: self.top_n]
        ranked = [(sym, gap) for sym, gap in ranked if gap > 0]
        if not ranked:
            return Signal(information_available_at=last_date)
        weight = 1.0 / len(ranked)
        return Signal(information_available_at=last_date, weights={s: weight for s, _ in ranked})
