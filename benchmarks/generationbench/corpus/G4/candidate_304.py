from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion50d(Strategy):
    rationale = (
        "This strategy exploits mean-reverting behavior in stock prices around specific "
        "price levels. By identifying stocks that have deviated significantly from their 50-day "
        "moving averages and expecting a reversion to the mean, we aim to capture profits from "
        "price movements that revert back towards historical norms."
    )

    def __init__(self, lookback: int = 50, top_n: int = 20) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._lookback).select(
            pl.col("symbol"), pl.col("close").alias("c")
        )

        moving_averages: pl.DataFrame = (
            history.groupby("symbol")
                   .agg(pl.col("adj_close").mean().alias("ma"))
                   .with_columns(pl.col("ma").shift(-1).alias("prev_ma"))
        )

        ranked_stocks = closes.join(moving_averages, on="symbol", how="inner")

        if "c" not in ranked_stocks.columns or "ma" not in ranked_stocks.columns:
            return Signal(information_available_at=stamp, weights={})

        deviations = (ranked_stocks["c"] - ranked_stocks["ma"]).abs().alias("dev")
        ranked_stocks = ranked_stocks.sort(by="dev", descending=True)

        long_picks: list[str] = []
        short_picks: list[str] = []

        for symbol in view.symbols:
            if symbol not in ranked_stocks.columns:
                continue
            latest_deviation = float(ranked_stocks[f"{symbol}_dev"].last())
            if latest_deviation > 0.1 * ranked_stocks["ma"][ranked_stocks["symbol"] == symbol].mean():
                long_picks.append(symbol)
            elif latest_deviation < -0.1 * ranked_stocks["ma"][ranked_stocks["symbol"] == symbol].mean():
                short_picks.append(symbol)

        picks = long_picks + short_picks
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight for s in picks[: self._top_n]
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest