from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class EarningsSentimentStrategy(Strategy):
    rationale = (
        "This strategy exploits the interplay between earnings surprises and social media sentiment. "
        "Earnings discrepancies from expected values and real-time market mood can offer mean reversion "
        "opportunities in stock prices."
    )

    def __init__(self, window: int = 30, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Load earnings data and social media sentiment scores
        earning_surprises = load_earnings_data()
        social_media_sentiment = load_social_media_data()

        # Filter to include only symbols present in both datasets
        common_symbols = list(set(earning_surprises.columns) & set(social_media_sentiment.columns))
        filtered_closes = closes.select([pl.col("symbol").filter(pl.col("symbol").is_in(common_symbols)), pl.col("adj_close")])

        # Calculate earnings surprise for each symbol
        earning_surprises["surprise"] = (earning_surprises["actual_eps"] - earning_surprises["expected_eps"]) / earning_surprises["expected_eps"]

        # Rank symbols based on composite score
        ranked_scores = []
        for symbol in common_symbols:
            if symbol not in filtered_closes.columns or symbol not in earning_surprises.columns:
                continue

            close_values = [float(v) for v in filtered_closes[symbol].drop_nulls().to_list()]
            if len(close_values) < self._window:
                continue

            surprise_value = float(earning_surprises[earning_surprises["symbol"] == symbol]["surprise"])
            sentiment_score = social_media_sentiment[social_media_sentiment["symbol"] == symbol]["sentiment"].mean()
            composite_score = (abs(surprise_value) + abs(sentiment_score)) / 2

            ranked_scores.append((symbol, surprise_value, sentiment_score, composite_score))

        # Sort and select top N symbols
        ranked_scores.sort(key=lambda x: -x[3])
        top_symbols = [score[0] for score in ranked_scores[:self._top_n]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest

def load_earnings_data() -> pl.DataFrame:
    # Placeholder function to load earnings data
    symbols = view.symbols
    return pl.DataFrame({
        "symbol": symbols,
        "actual_eps": [0.5, 1.2, -0.3] * (len(symbols) // 3),
        "expected_eps": [0.4, 1.1, -0.2] * (len(symbols) // 3)
    })

def load_social_media_data() -> pl.DataFrame:
    # Placeholder function to load social media data
    symbols = view.symbols
    return pl.DataFrame({
        "symbol": symbols,
        "sentiment": [0.8, -0.7, 1.0] * (len(symbols) // 3)
    })