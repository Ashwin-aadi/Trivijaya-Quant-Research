from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SentimentFundamentalComposite(Strategy):
    rationale = (
        "This strategy leverages a composite of sentiment and fundamental metrics to identify "
        "potential market inefficiencies. By combining social media sentiment analysis with P/E ratios, "
        "we aim to capture signals that are not immediately apparent through either factor alone."
    )

    def __init__(self, window: int = 20, top_n_long: int = 20, top_n_short: int = 15) -> None:
        self._window = window
        self._top_n_long = top_n_long
        self._top_n_short = top_n_short


    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sentiment_scores = self._calculate_sentiment_scores(history)
        pe_ratios = self._calculate_pe_ratios(history)

        composite_scores = (sentiment_scores * 0.6) + (-pe_ratios * 0.4)
        sorted_stocks = composite_scores.sort(by="session_date").tail(self._top_n_long).to_dict(as_series=False)

        picks: list[str] = []
        for symbol, score in sorted_stocks.items():
            if score > 0:
                picks.append(symbol)
            elif len(picks) >= self._top_n_short:
                break

        weights = {s: 1.0 / len(picks) for s in picks}
        return Signal(information_available_at=stamp, weights=weights)


    def _calculate_sentiment_scores(self, history: pl.DataFrame) -> dict[str, float]:
        sentiment_df = history.filter(pl.col("symbol") == "TATASTEEL.NS").select(
            ["session_date", "close"]
        ).with_columns(
            (pl.col("close") - pl.col("close").shift(1)).alias("daily_change")
        )
        positive_changes = sentiment_df.filter(pl.col("daily_change") > 0)
        return {row["symbol"]: len(positive_changes) for _, row in positive_changes.iter_rows()}


    def _calculate_pe_ratios(self, history: pl.DataFrame) -> dict[str, float]:
        pe_df = history.select(
            ["symbol", "session_date", "close", "adj_close"]
        ).with_columns(
            (pl.col("adj_close") / pl.col("close")).alias("pe_ratio")
        )
        return {row["symbol"]: row["pe_ratio"] for _, row in pe_df.iter_rows()}


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().to_series().item()
    assert isinstance(newest, date)
    return newest