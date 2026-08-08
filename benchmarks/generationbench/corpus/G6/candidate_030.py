from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RSIRelativeStrength(Strategy):
    rationale = (
        "This strategy identifies stocks with high relative strength compared to the NIFTY 50 index "
        "by leveraging the Relative Strength Index (RSI). High RSI values indicate strong performance, "
        "while a stop-loss mechanism ensures risk is managed."
    )

    def __init__(self, window: int = 14, threshold_buy: float = 85.0, threshold_sell: float = 70.0, n_stocks: int = 30) -> None:
        self._window = window
        self._threshold_buy = threshold_buy
        self._threshold_sell = threshold_sell
        self._n_stocks = n_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate RSI for each stock
        def calc_rsi(df: pl.DataFrame) -> float:
            delta = df.select(
                (pl.col("close") - pl.col("open")).alias("delta")
            )
            gain = delta.with_column(pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0))
            loss = delta.with_column(pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0))

            avg_gain = (
                gain.groupby("symbol").agg(
                    (pl.col("delta").sum() / pl.lit(self._window)).alias("avg_gain")
                )
            )
            avg_loss = (
                loss.groupby("symbol").agg(
                    (pl.col("delta").sum() / pl.lit(self._window)).alias("avg_loss")
                )
            )

            rs = avg_gain.join(avg_loss, on="symbol", how="inner").select(
                ((pl.col("avg_gain") / pl.col("avg_loss")).fill_null(0).alias("rs"))
            ).with_column(
                (100 - 100 * (1 + pl.col("rs")).rank(method="average", descending=True)).alias("rsi")
            )

            return rs["rsi"][0]

        nifty50_rsi = calc_rsi(history.filter(pl.col("symbol").is_in(view.symbols[:50])))
        stock_rsies = history.select(
            pl.col("symbol"),
            (pl.col("close") - pl.col("open")).alias("delta")
        ).with_column(
            ((pl.col("delta").sum() / pl.lit(self._window)).alias("avg_gain"))
        ).with_column(
            (((pl.col("delta").shift(1).rolling_sum(2) - pl.col("open")) / 2).alias("avg_loss"))
        ).group_by("symbol").agg(
            (100 * (pl.col("avg_gain") / (pl.col("avg_gain") + pl.col("avg_loss")).fill_null(1)).rank(method="average", descending=True)).alias("rsi")
        )

        above_nifty = stock_rsies.filter(pl.col("rsi") > self._threshold_buy - 15)
        strong_stocks = above_nifty.sort("rsi", descending=True).head(self._n_stocks)

        if strong_stocks.height < self._n_stocks:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [s[0] for s in strong_stocks.to_dict(False)["symbol"]]
        weight = 1.0 / len(picks)
        signal = Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )

        # Ensure stop-loss mechanism
        if (view.closes(lookback=self._window) > view.closes()).any():
            return Signal(information_available_at=stamp, weights={})

        return signal


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest