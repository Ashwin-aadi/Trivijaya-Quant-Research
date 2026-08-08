from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SentimentEarningsStrategy(Strategy):
    rationale = (
        "This strategy leverages a combination of social media sentiment and earnings "
        "surprise to identify mispriced stocks. Positive sentiment is expected to precede "
        "strong earnings reports, potentially leading to arbitrage opportunities."
    )

    def __init__(self, window_sentiment: int = 5, threshold_surprise: float = 0.1) -> None:
        self._window_sentiment = window_sentiment
        self._threshold_surprise = threshold_surprise

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_sentiment + 20)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sentiment_scores = self._calculate_sentiment_scores(history)
        earnings_surprises = self._calculate_earnings_surprises(history)

        combined_scores = pl.DataFrame({
            "symbol": history["symbol"],
            "sentiment_score": [float(sentiment_scores.get(s, 0.0)) for s in history["symbol"]],
            "earnings Surprise": [float(earnings_surprises.get(s, 0.0)) for s in history["symbol"]]
        })

        combined_scores = (
            combined_scores
            .sort(["sentiment_score", "earnings Surprise"], descending=[True, True])
            .head(self._window_sentiment)
        )

        selected_symbols = [
            symbol for _, (symbol, sentiment_score, earnings_surprise) in
            enumerate(combined_scores.iter_rows())
            if sentiment_score > 0 and earnings_surprise > self._threshold_surprise
        ]

        weights = {s: 1.0 / len(selected_symbols) for s in selected_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_sentiment_scores(history: pl.DataFrame) -> dict[str, float]:
    sentiment_history = history.filter(pl.col("symbol").is_in(view.symbols))
    sentiment_avg_scores = (
        sentiment_history
        .group_by("symbol")
        .agg((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("sentiment_score"))
        .sort("sentiment_score", descending=True)
    )
    return {row["symbol"]: float(row["sentiment_score"]) for row in sentiment_avg_scores.iter_rows()}


def _calculate_earnings_surprises(history: pl.DataFrame) -> dict[str, float]:
    earnings_history = history.filter(pl.col("symbol").is_in(view.symbols))
    earnings_avg_scores = (
        earnings_history
        .group_by("symbol")
        .agg((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("earnings Surprise"))
        .sort("earnings Surprise", descending=True)
    )
    return {row["symbol"]: float(row["earnings Surprise"]) for row in earnings_avg_scores.iter_rows()}