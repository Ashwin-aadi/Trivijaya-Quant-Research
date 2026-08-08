from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthIndexStrategy(Strategy):
    rationale = (
        "This strategy identifies stocks with a higher relative strength index (RSI) compared to the broader Nifty 50 index. "
        "By focusing on outperforming stocks relative to the market, we aim to capture long-term outperformance driven by positive market sentiment and fundamental strengths."
    )

    def __init__(self, window: int = 14, threshold: float = 70, max_positions: int = 20) -> None:
        self._window = window
        self._threshold = threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        nifty_50_history = history.filter(pl.col("symbol") == "NIFTY 50")

        if nifty_50_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_50_closes = nifty_50_history.select(
            pl.col("close").alias("nifty_close")
        ).to_vertical()

        all_closes = view.closes(lookback=self._window)
        combined_df = (
            history.join(all_closes, on="session_date", how="inner")
                .with_column(pl.col("symbol").cast(pl.Categorical))
                .select(
                    pl.col("symbol"),
                    "close",
                    nifty_50_closes["nifty_close"],
                )
        )

        rsi = _compute_rsi(combined_df)
        combined_df = combined_df.join(rsi, on="symbol", how="inner")

        filtered_stocks = combined_df.filter(
            (pl.col("rsi") > self._threshold) & (pl.col("close") / pl.col("nifty_close") > 1.05)
        )

        if filtered_stocks.height < self._max_positions:
            return Signal(information_available_at=stamp, weights={})

        selected_symbols = [
            s for _, s in filtered_sticks.sort("rsi", descending=True).head(self._max_positions)["symbol"]
        ]
        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_rsi(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("return"),
        ((pl.col("high") + pl.col("low")) / 2 - pl.col("close").shift(1)) / 2.0
    ).with_column(
        pl.when(pl.col("return").is_nan()).then(0).otherwise(pl.col("return")).alias("adjusted_return")
    )

    gain = df.with_columns((pl.col("adjusted_return") * (pl.col("adjust_return") > 0)).alias("gain"))
    loss = df.with_columns((-pl.col("adjusted_return") * (pl.col("adjust_return") < 0)).alias("loss"))

    avg_gain = (
        gain.groupby("symbol").agg(
            ((pl.col("gain").sum() / self._window).alias("avg_gain"))
        )
    )

    avg_loss = (
        loss.groupby("symbol").agg(
            ((pl.col("loss").sum() / self._window).alias("avg_loss"))
        )
    )

    rs = (avg_gain["avg_gain"] / avg_loss["avg_loss"]).alias("rs")

    rsi = 100 - (100 / (1 + rs))
    return df.join(rsi, on="symbol", how="inner")