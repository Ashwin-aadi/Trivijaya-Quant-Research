from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SentimentMacroeconomicComposite(Strategy):
    rationale = (
        "This strategy leverages discrepancies between social media sentiment and macroeconomic indicators "
        "to identify potential mispricings in the Indian market. By combining these signals, we aim to capitalize on "
        "lagging price adjustments relative to real-time sentiment or unexpected macro shifts."
    )

    def __init__(self, window_sentiment: int = 7, window_macro: int = 30, weight_sentiment: float = 0.6, weight_macro: float = 0.4) -> None:
        self._window_sentiment = window_sentiment
        self._window_macro = window_macro
        self._weight_sentiment = weight_sentiment
        self._weight_macro = weight_macro

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_sentiment + self._window_macro)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sentiment_scores = self._calculate_sentiment_scores(history, self._window_sentiment)
        macro_scores = self._calculate_macroeconomic_scores(history, self._window_macro)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in sentiment_scores or symbol not in macro_scores:
                continue
            composite_score = (sentiment_scores[symbol] * self._weight_sentiment) + (macro_scores[symbol] * self._weight_macro)
            picks.append(symbol)

        picks = picks[:20]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )

    def _calculate_sentiment_scores(self, history: pl.DataFrame, window: int) -> dict[str, float]:
        sentiment_history = history.filter(pl.col("symbol").is_in(view.symbols)).select([
            "symbol", pl.col("session_date"), (pl.col("adj_close") - pl.col("adj_close").shift(1)) / pl.col("adj_close").shift(1).alias("return")
        ])
        return sentiment_history.groupby("symbol").agg(
            pl.col("return").mean().alias("sentiment_score")
        ).collect().with_columns(pl.col("sentiment_score").rank(method="ordinal", descending=True)).to_dict(as_pandas=False)["sentiment_score"]

    def _calculate_macroeconomic_scores(self, history: pl.DataFrame, window: int) -> dict[str, float]:
        macro_history = history.select([
            "session_date", (pl.col("adj_close") - pl.col("adj_close").shift(1)) / pl.col("adj_close").shift(1).alias("return")
        ])
        return macro_history.groupby("session_date").agg(
            pl.col("return").mean().alias("macro_score")
        ).collect().with_columns(pl.col("macro_score").rank(method="ordinal", descending=True)).to_dict(as_pandas=False)["macro_score"]

def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest