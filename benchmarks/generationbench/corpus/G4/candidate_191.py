from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class EarningsSentimentComposite(Strategy):
    rationale = (
        "This strategy exploits a combination of earnings quality and analyst sentiment to "
        "identify potential mispricings. By combining these two weakly related characteristics, "
        "we aim to capture opportunities that may arise from either strong earnings or positive"
        "sentiment, even when the other characteristic is weak."
    )

    def __init__(self, window_earnings: int = 365, window_sentiment: int = 120) -> None:
        self._window_earnings = window_earnings
        self._window_sentiment = window_sentiment

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window_earnings, self._window_sentiment))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        earnings_quality_scores = self._compute_earnings_quality(history)
        sentiment_scores = self._compute_sentiment(history)

        if earnings_quality_scores.height < 1 or sentiment_scores.height < 1:
            return Signal(information_available_at=stamp, weights={})

        composite_scores = (
            earnings_quality_scores.with_columns(sentiment_scores.alias("sentiment_score"))
                .with_column(
                    (pl.col("earnings_quality") + pl.col("sentiment_score")) / 2.0
                )
                .select(pl.col("symbol"), pl.last("earnings_quality").alias("composite_score"))
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in composite_scores["symbol"].to_list():
                continue
            score = float(composite_scores.filter(pl.col("symbol") == symbol)["composite_score"][0])
            picks.append(symbol)

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_earnings_quality(history: pl.DataFrame) -> pl.DataFrame:
    symbols = history.select(pl.col("symbol")).unique().to_series().to_list()
    accruals_ratio = (
        view.history(lookback=365).select(
            pl.col("symbol"),
            (pl.col("net_income") - pl.col("cash_flow_from_operating_activities")) / pl.col("total_assets").alias("accruals_ratio")
        ).filter(pl.col("session_date") > date.today() - 365)
    )
    return accruals_ratio.select(
        pl.col("symbol"), (pl.col("accruals_ratio") + 1).rank(method="ordinal", descending=True).alias("earnings_quality")
    )


def _compute_sentiment(history: pl.DataFrame) -> pl.DataFrame:
    symbols = history.select(pl.col("symbol")).unique().to_series().to_list()
    sentiment = (
        view.history(lookback=365).select(
            pl.col("symbol"), (pl.col("rating") + 1).rank(method="ordinal", descending=True).alias("sentiment_score")
        ).filter(pl.col("session_date") > date.today() - 365)
    )
    return sentiment