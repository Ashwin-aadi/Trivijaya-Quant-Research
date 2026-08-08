from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SentimentPEStrategy(Strategy):
    rationale = (
        "This strategy combines social media sentiment analysis with P/E ratio to identify "
        "potentially undervalued stocks. Negative sentiment and low P/E ratios suggest market "
        "overreactions that may correct over time."
    )

    def __init__(self, window: int = 30, p_e_threshold: float = 20.0, top_n: int = 20) -> None:
        self._window = window
        self._p_e_threshold = p_e_threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sentiment_scores = _compute_sentiment_scores(view)
        p_e_ratios = _compute_pe_ratios(view)

        if sentiment_scores.is_empty() or p_e_ratios.is_empty():
            return Signal(information_available_at=stamp, weights={})

        combined_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in sentiment_scores.columns or symbol not in p_e_ratios.columns:
                continue
            score = (1 - sentiment_scores[symbol].mean()) + (p_e_ratios[symbol].median() / self._p_e_threshold)
            combined_scores[symbol] = score

        ranked_symbols = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        picks = [s for s, _ in ranked_symbols[:self._top_n]]
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


def _compute_sentiment_scores(view: MarketView) -> pl.DataFrame:
    history = view.history(lookback=30)
    symbols = [symbol for symbol in view.symbols if f"sentiment_{symbol}" in history.columns]
    sentiment_data = history.select(["session_date"] + symbols).sort("session_date")
    return sentiment_data.groupby("symbol").agg(
        pl.col(f"sentiment_{symbol}").mean().alias(f"avg_sentiment_{symbol}")
    ).with_columns(
        (1 - pl.col(f"avg_sentiment_{symbol}")).alias(f"score_{symbol}") for symbol in symbols
    )


def _compute_pe_ratios(view: MarketView) -> pl.DataFrame:
    history = view.history(lookback=365)
    closes = [close for close in view.closes() if f"pe_{close.symbol}" in close.columns]
    pe_data = pl.concat(closes).select(
        ["session_date", "symbol"] + [f"pe_{close.symbol}" for close in closes]
    ).sort("session_date")
    return pe_data.groupby("symbol").agg(
        pl.col(f"pe_{symbol}").median().alias(f"median_pe_{symbol}")
    )