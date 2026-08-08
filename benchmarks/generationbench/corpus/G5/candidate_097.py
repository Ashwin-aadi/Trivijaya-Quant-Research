from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of stock prices to revert to their mean "
        "over short time horizons (e.g., 10 days). By identifying stocks that have deviated significantly from "
        "their recent average price, we can capture potential reversals."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.select(pl.col("adj_close").mean())
            .collect()
            .with_column(pl.lit(symbols).alias("symbol"))
            .rename({"col_0": "mean"})
            .to_dict(as_series=False)
        )

        thresholded = _threshold_mean_reversion(history, mean_close, self._threshold)

        picks: list[str] = []
        for symbol in symbols:
            if f"{symbol}_score" not in thresholded.columns:
                continue
            score = float(thresholded[f"{symbol}_score"].to_list()[0])
            if score > 0:
                picks.append(symbol)
        
        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

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


def _threshold_mean_reversion(history: pl.DataFrame, mean_close: dict[str, float], threshold: float) -> pl.DataFrame:
    history = history.join(pl.DataFrame(mean_close), on="symbol", how="inner")
    history = (
        history.with_columns(
            (pl.col("adj_close") - pl.col("mean")).abs().alias("deviation"),
            (((pl.col("adj_close") - pl.col("mean")) / pl.col("mean")).abs() * 100).alias("percent_deviation"),
        )
    ).with_column(pl.when((pl.col("deviation") > threshold) | (pl.col("percent_deviation") > threshold)).then(1).otherwise(0).alias("score"))

    return history