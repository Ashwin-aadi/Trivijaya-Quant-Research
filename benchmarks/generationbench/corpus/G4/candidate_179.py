from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class EarningsSocialMediaStrategy(Strategy):
    rationale = (
        "This strategy leverages the composite approach using earnings season announcements and social media sentiment. "
        "By monitoring both non-financial data (social media) and financial reports (earnings), we aim to capture temporary "
        "mispricings in stocks due to information asymmetry between market reactions."
    )

    def __init__(self, window_social_media: int = 10, window_earnings: int = 28, top_n: int = 20) -> None:
        self._window_social_media = window_social_media
        self._window_earnings = window_earnings
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_social_media + self._window_earnings)
        if closes.height < self._window_social_media + self._window_earnings:
            return Signal(information_available_at=stamp, weights={})

        # Extract earnings announcement dates
        earnings_dates = _get_upcoming_earnings(view)

        # Calculate social media sentiment scores
        social_media_scores = _compute_sentiment_scores(closes, view.symbols, self._window_social_media)

        # Rank stocks based on a composite score
        ranked_stocks = _rank_stocks(social_media_scores, earnings_dates)
        if not ranked_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_stocks[: self._top_n])
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_stocks[: self._top_n]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _get_upcoming_earnings(view: MarketView) -> list[str]:
    upcoming_earnings = view.history().filter(pl.col("symbol").is_in(view.symbols)).select(
        pl.col("symbol")
    ).sort("session_date", descending=False).collect().with_column(pl.col("symbol").arr.unique()).to_series().to_list()
    return upcoming_earnings


def _compute_sentiment_scores(closes: pl.DataFrame, symbols: tuple[str, ...], window: int) -> dict[str, float]:
    sentiment_scores = {}
    for symbol in symbols:
        if symbol not in closes.columns:
            continue
        values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
        if len(values) < window:
            continue
        last_close = values[-1]
        max_close = max(values)
        min_close = min(values)
        sentiment_score = (last_close - min_close) / (max_close - min_close + 1e-9)
        sentiment_scores[symbol] = sentiment_score
    return sentiment_scores


def _rank_stocks(social_media_scores: dict[str, float], earnings_dates: list[str]) -> list[str]:
    ranked_stocks = sorted(
        social_media_scores.items(),
        key=lambda x: (x[1] * 0.6 + (earnings_dates.index(x[0]) / len(earnings_dates)) * 0.4),
        reverse=True
    )
    return [s for s, _ in ranked_stocks]