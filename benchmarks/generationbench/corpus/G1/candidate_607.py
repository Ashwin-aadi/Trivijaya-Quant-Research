from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates increased market volatility and reduced trading ranges. "
        "During such periods, stocks may experience price fluctuations that can provide "
        "profitable entry points for investors."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        compression_scores: dict[str, float] = {}
        for symbol in symbols:
            high_low_diff = (
                pl.col("high").max().cast(pl.Float64) - pl.col("low").min().cast(pl.Float64)
            ).alias("range")
            mean_close = pl.col("close").mean().alias("mean")
            compression_score = (history.select([symbol, "session_date", high_low_diff, mean_close])["range"] / history.select([symbol, "session_date", high_low_diff, mean_close])["mean"]).sum()
            if not compression_score.is_null():
                compression_scores[symbol] = float(compression_score)

        top_symbols = sorted(
            compression_scores.items(), key=lambda item: item[1], reverse=True
        )[:5]
        weights = {symbol: 0.2 for symbol, _ in top_symbols}
        return Signal(information_available_at=stamp, weights={s: weight for s, weight in weights.items() if not pl.DataFrame(weights).is_empty()})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest