from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "This strategy exploits relative strength (RS) by identifying stocks in the Indian "
        "market that outperform the Nifty 50 index. By buying into these stocks and selling those "
        "that underperform, we aim to capture temporary mispricings and momentum."
    )

    def __init__(self, window: int = 60, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_50_close = view.closes().select(pl.col("NIFTY50").alias("close"))
        stock_closes = history.select(
            pl.col(view.symbols).alias(*view.symbols)
        ).transpose().rename({0: "close", 1: "symbol"})

        returns = (
            (history["close"] / history["close"].shift(1) - 1.0)
            .with_column(history["session_date"])
            .select(pl.col("symbol"), pl.col("session_date").alias("date"), pl.all().exclude("date"))
        )

        nifty_50_returns = (
            (nifty_50_close["close"] / nifty_50_close["close"].shift(1) - 1.0)
            .with_column(nifty_50_close["session_date"])
            .select(pl.col("symbol"), pl.col("session_date").alias("date"), pl.all().exclude("date"))
        )

        combined = (
            returns.join(stock_closes, on=["symbol", "date"], how="inner")
            .join(nifty_50_returns, on=["date"], how="inner")
            .select(pl.col("symbol"), pl.col("close").sum().alias("stock_return"),
                    pl.col("close_x").sum().alias("nifty_return"))
        )

        combined = (
            combined.with_columns(
                (combined["stock_return"] / combined["nifty_return"]).alias("rs_score")
            )
            .sort("rs_score", descending=True)
            .head(20)["symbol"]
        ).to_list()

        weight = 1.0 / len(combined) if combined else 0.0

        rs_scores = [
            (float(view.closes().select(pl.col(symbol).last().alias("close")).row(0)[0] /
                   view.closes().select(pl.col("NIFTY50").last().alias("close")).row(0)[0]) - 1) * weight
            for symbol in combined if float(view.closes().select(pl.col(symbol).last().alias("close")).row(0)[0])
        ]

        return Signal(
            information_available_at=stamp,
            weights=dict(zip(combined, rs_scores))
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest