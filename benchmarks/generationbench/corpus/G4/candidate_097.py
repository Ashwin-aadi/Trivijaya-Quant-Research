from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion in the Indian equity market by "
        "identifying stocks that deviate from their historical price averages. Stocks showing "
        "overbought or oversold conditions based on Bollinger Bands and RSI are selected for "
        "short-term trades, with the goal of profiting from price reversions."
    )

    def __init__(self, window: int = 20, rsi_window: int = 14) -> None:
        self._window = window
        self._rsi_window = rsi_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Compute Bollinger Bands
        bb_lower = closes.with_columns(
            (
                (pl.col("adj_close").rolling_mean(self._window) - 2 * pl.col("adj_close").rolling_std(self._window))
                .alias("bb_lower")
            )
        )
        bb_upper = closes.with_columns(
            (
                (pl.col("adj_close").rolling_mean(self._window) + 2 * pl.col("adj_close").rolling_std(self._window))
                .alias("bb_upper")
            )
        )

        # Compute RSI
        rsi = _compute_rsi(closes, self._rsi_window)

        # Identify overbought and oversold stocks based on Bollinger Bands and RSI
        candidates: list[str] = []
        for symbol in view.symbols:
            if (
                (float(bb_upper[symbol].to_list()[-1]) > float(closes[symbol].to_list()[-1]))
                or (float(bb_lower[symbol].to_list()[-1]) < float(closes[symbol].to_list()[-1]))
                or (float(rsi[symbol].to_list()[-1]) > 70.0)
                or (float(rsi[symbol].to_list()[-1]) < 30.0)
            ):
                candidates.append(symbol)

        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        # Rank and select top N candidates
        ranks = [i for i, c in enumerate(candidates) if (c in rsi.columns)]
        weighted_ranks = sorted(zip(ranks, candidates), key=lambda x: x[0])
        picks = [s for _, s in weighted_ranks[:25]]

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


def _compute_rsi(closes: pl.DataFrame, window: int) -> pl.DataFrame:
    delta = closes.with_columns(
        (
            (pl.col("adj_close").diff().abs()).alias("delta")
        )
    )
    gain = delta.with_columns(
        ((pl.col("delta") / 2).alias("gain"))
    )
    loss = delta.with_columns(
        ((-1 * pl.col("delta")).alias("loss"))
    )

    avg_gain = gain.with_columns(
        (pl.col("gain").rolling_mean(window)).alias("avg_gain")
    )
    avg_loss = loss.with_columns(
        (pl.col("loss").rolling_mean(window)).alias("avg_loss")
    )

    rs = avg_gain.join(avg_loss, on="symbol", how="outer").with_columns(
        (
            (100 - 100 * pl.col("avg_gain") / (pl.col("avg_gain") + pl.col("avg_loss"))).alias("rs")
        )
    ).select("symbol", "rs")

    return rs