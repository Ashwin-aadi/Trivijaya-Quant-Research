from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionAndDispersion(Strategy):
    rationale = (
        "This strategy exploits the phenomenon of stock price dispersion and range compression. "
        "High dispersion often precedes mean reversion, while low range compression signals potential "
        "mean reversion opportunities. By identifying stocks with high dispersion or low range compression, "
        "we aim to capture these mean reversion events."
    )

    def __init__(self, window: int = 30, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=60)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_low_diff = (
            history.select(
                pl.col("high") - pl.col("low").alias("high_low_diff"),
                (pl.col("high") - pl.col("low")).rolling_mean(window_size=self._window).alias(f"mean_high_low_{self._window}"),
                (pl.col("high") / pl.col("low") - 1.0).rolling_mean(window_size=self._window).alias(f"mean_range_ratio_{self._window}")
            )
        )

        if high_low_diff.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        dispersion_scores = (
            high_low_diff
            .with_column(
                (pl.col("high_low_diff") / pl.col(f"mean_high_low_{self._window}")).alias("dispersion_score")
            )
            .sort("session_date", descending=False)
            .reverse()
            .select(
                pl.col("symbol"),
                pl.col("high_low_diff"),
                pl.col(f"mean_range_ratio_{self._window}"),
                (pl.col("adj_close").rank(method="max", descending=True) / 60 * 100).alias("volatility_rank")
            )
        )

        high_dispersion_stocks = (
            dispersion_scores
            .sort("dispersion_score", descending=True)
            .head(self._top_n)
            .select("symbol")
            .to_dict(False)
        )

        low_range_compression_stocks = (
            dispersion_scores
            .filter(pl.col(f"mean_range_ratio_{self._window}") < 0.25 * pl.col(f"mean_range_ratio_{self._window}").max())
            .sort("volatility_rank", descending=True)
            .head(self._top_n)
            .select("symbol")
            .to_dict(False)
        )

        stocks = high_dispersion_stocks + low_range_compression_stocks
        if not stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest