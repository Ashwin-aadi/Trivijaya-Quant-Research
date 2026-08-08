from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class EarningsSentiment(Strategy):
    rationale = (
        "This strategy leverages a composite of earnings surprises (ES) and social media sentiment analysis (SMA) "
        "to identify stocks with both positive financial performance indications and favorable public perception."
    )

    def __init__(self, es_window: int = 20, sma_window: int = 30, top_n: int = 15) -> None:
        self._es_window = es_window
        self._sma_window = sma_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._es_window + self._sma_window)
        if closes.height < self._es_window + self._sma_window:
            return Signal(information_available_at=stamp, weights={})

        es_scores = self._compute_es_scores(closes)
        sma_scores = self._compute_sma_scores(view)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in es_scores or symbol not in sma_scores:
                continue
            rank_es = es_scores[symbol]
            rank_sma = sma_scores[symbol]
            if rank_es is None or rank_sma is None:
                continue

            combined_score = (rank_es * 2 + rank_sma) / 3
            picks.append((symbol, combined_score))

        picks.sort(key=lambda x: x[1], reverse=True)
        picks = [p for p in picks if p[1] > 0][: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(weight for _, weight in picks)
        weights = {symbol: weight / total_weight for symbol, weight in picks}
        return Signal(
            information_available_at=stamp,
            weights={symbol: value for symbol, value in weights.items()},
        )

    def _compute_es_scores(self, closes: pl.DataFrame) -> dict[str, float]:
        es_scores: dict[str, float] = {}
        history = view.history(lookback=self._es_window)
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in closes.columns:
                continue
            earnings_history = history[symbol]
            adj_close_history = closes[symbol].drop_nulls().to_list()
            if len(adj_close_history) < self._es_window:
                continue

            actual_earnings = float(earnings_history[-1])
            consensus_estimates = [
                float(v)
                for v in earnings_history.filter(pl.col("session_date") > date.today() - pl.duration(days=self._es_window))
                .sort("session_date")
                .head(1)["consensus_estimate"]
                if not pl.col("consensus_estimate").is_null()
            ]
            if consensus_estimates:
                consensus = sum(consensus_estimates) / len(consensus_estimates)
                es_score = actual_earnings - consensus
                if es_score > 0:
                    es_scores[symbol] = es_score

        return {k: v for k, v in sorted(es_scores.items(), key=lambda item: item[1], reverse=True)}

    def _compute_sma_scores(self, view: MarketView) -> dict[str, float]:
        sma_scores: dict[str, float] = {}
        history = view.history(lookback=self._sma_window)
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            sma_score = 0.0
            sentiment_posts = [
                post
                for post in history[history["symbol"] == symbol]["text"]
                .to_list()
                if "social_media" in post
            ]
            if sentiment_posts:
                positive_count, negative_count = 0, 0
                for post in sentiment_posts[-self._sma_window:]:
                    if "positive" in post:
                        positive_count += 1
                    elif "negative" in post:
                        negative_count += 1

                sma_score = (positive_count - negative_count) / len(sentiment_posts)
                sma_scores[symbol] = sma_score

        return {k: v for k, v in sorted(sma_scores.items(), key=lambda item: item[1], reverse=True)}


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest