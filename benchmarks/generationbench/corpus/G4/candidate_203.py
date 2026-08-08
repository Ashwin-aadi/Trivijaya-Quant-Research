from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy

class EarningsSurpriseSentiment(Strategy):
    rationale = (
        "This strategy combines earnings surprise with social media sentiment to identify "
        "stocks presenting strong composite signals. Earnings surprises can indicate underlying "
        "company-specific risks or opportunities, while social media sentiment captures real-time "
        "market psychology. By integrating these two, the strategy aims to capitalize on both "
        "company-specific anomalies and broader market sentiments."
    )

    def __init__(self, earnings_window: int = 1, sentiment_window: int = 7, top_n: int = 30) -> None:
        self._earnings_window = earnings_window
        self._sentiment_window = sentiment_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._earnings_window + self._sentiment_window - 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        earnings = history.select([
            pl.col("symbol"),
            (pl.col("close") / pl.col("open").shift(self._earnings_window) - 1).alias("earnings_surprise")
        ])
        sentiment = _calculate_sentiment(view.closes(lookback=self._sentiment_window), self._top_n)

        combined = earnings.join(sentiment, on="symbol", how="inner")

        if combined.is_empty():
            return Signal(information_available_at=stamp, weights={})

        combined = combined.with_columns(
            (pl.col("earnings_surprise") + pl.col("sentiment_score")).alias("composite_score")
        )
        sorted_stocks = combined.sort("composite_score", descending=True).select("symbol")

        picks = [row["symbol"] for row in sorted_stocks.to_dict(orient="records")[:self._top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )

def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest

def _calculate_sentiment(closes: pl.DataFrame, top_n: int) -> pl.DataFrame:
    sentiment_scores = {}
    for symbol in closes.columns[1:]:
        values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
        if len(values) >= 7:
            avg_sentiment = sum(values[-7:]) / 7
            sentiment_scores[symbol] = avg_sentiment

    sorted_scores = sorted(sentiment_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    top_symbols = [symbol for symbol, score in sorted_scores[:top_n]]
    scores_df = pl.DataFrame({"symbol": top_symbols})
    
    if not scores_df.is_empty():
        scores_df = scores_df.with_columns(pl.Series(name="sentiment_score", values=[score for _, score in sorted_scores[:top_n]]))

    return scores_df