from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumStrategy(Strategy):
    rationale = (
        "This strategy selects stocks based on their momentum relative to the Nifty 100 index and "
        "their RSI over the past 10 trading days. High-momentum stocks are selected for inclusion in a "
        "portfolio of top 25 stocks, with equal weights assigned to each stock."
    )

    def __init__(self, window: int = 60, rsi_window: int = 10) -> None:
        self._window = window
        self._rsi_window = rsi_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._rsi_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_history = history.filter(pl.col("symbol") == "NIFTY 100")
        stock_history = history.select(
            [pl.col("symbol"), pl.col("session_date"), pl.col("close")]
        )

        # Calculate returns relative to Nifty
        nifty_returns = (
            nifty_history.lazy()
            .select((pl.col("adj_close").shift(-self._window) / pl.col("adj_close") - 1.0).alias("nifty_return"))
            .collect()
        )
        stock_returns = (
            stock_history.with_columns(
                (pl.col("close").shift(-self._window) / pl.col("close") - 1.0).alias("stock_return")
            ).select(["symbol", "session_date", "stock_return"])
        )

        # Calculate RSI
        def rsi(df: pl.DataFrame, window: int) -> pl.Series:
            delta = df["close"].diff().drop_nulls()
            gain = delta.where(delta > 0).mean().over(window)
            loss = -delta.where(delta < 0).mean().over(window)
            rs = gain / loss
            return 100.0 - (100.0 / (1 + rs))

        rsi_data = (
            stock_history.lazy()
            .with_columns(
                (pl.col("close").shift(-self._rsi_window) / pl.col("close") - 1.0).alias("price_change"),
                rsi(pl.select(stock_history, "symbol", "session_date", "adj_close"), self._rsi_window).alias("rsi"),
            )
            .collect()
        )

        # Join returns and RSI
        combined_data = (
            stock_returns.join(rsi_data, on=["symbol", "session_date"], how="inner")
            .join(nifty_returns, on="session_date", how="inner")
            .select(["symbol", "stock_return", "nifty_return", "rsi"])
        )

        # Filter and rank stocks
        ranked_stocks = (
            combined_data.with_columns(
                (pl.col("stock_return") - pl.col("nifty_return")).alias("momentum")
            )
            .sort("momentum", descending=True)
            .select(["symbol", "momentum", "rsi"])
        )

        # Select top 25 stocks
        picks = ranked_stocks.filter(pl.col("rsi").between(30, 70)).limit(25)

        if picks.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks["symbol"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest