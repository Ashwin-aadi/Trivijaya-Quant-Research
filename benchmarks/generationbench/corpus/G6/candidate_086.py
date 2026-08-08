from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to capture trends by scaling trades based on historical volatility. "
        "It enters positions when the security's closing price crosses above its 50-period moving average and volatility exceeds a threshold."
    )

    def __init__(self, window: int = 20, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 50)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        df = (
            pl.DataFrame({"symbol": symbols})
            .with_column(pl.col("symbol").cast(pl.Utf8))
            .join(
                how="inner",
                left_on="symbol",
                right=history.select(pl.all().exclude("session_date")),
                on="symbol",
            )
            .select(["symbol", "session_date", pl.all().except_("symbol")])
        )

        closes = df["adj_close"].to_list()
        mean_return = sum(closes[-50:] - closes[:-50]) / 50
        std_dev = (sum((x - mean_return) ** 2 for x in closes[-50:]) / 49) ** 0.5

        trends = (
            df.with_column(
                (pl.col("adj_close") - pl.col("adj_close").shift(50)) / 50
            )
            .select(["symbol", "session_date", (pl.col("adj_close") - pl.col("adj_close").shift(50)).alias("return")])
            .filter(
                (pl.col("return") > 1.5 * mean_return) &
                (pl.col("return") / std_dev > 3)
            )
        )

        if trends.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks = [symbol for symbol in trends["symbol"].to_list()][: self._top_n]
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