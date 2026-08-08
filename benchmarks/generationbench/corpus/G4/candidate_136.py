from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "This strategy exploits price-level reversion against a trailing reference. It "
        "identifies assets that have deviated significantly from their 20-day moving average "
        "and generates buy/sell signals accordingly."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(closes.columns) <= 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [col for col in closes.columns if col != "session_date"]
        sma = (
            closes.select(pl.col(symbols).mean())
            .with_columns(
                (pl.col(symbols) / pl.col("mean").alias("relative_price")).rowwise()
                .sum()
                .over("session_date")
                .alias("z_score")
            )
            .sort("session_date", descending=True)
            .head(1)["z_score"]
        )

        signal_weights = {}
        for symbol in symbols:
            z_score = float(sma[symbol])
            if abs(z_score) > 1.0:
                weight = (2.5 / len(symbols)) * (-1 if z_score < 0 else 1)
                signal_weights[symbol] = max(-0.03, min(0.03, weight))

        return Signal(information_available_at=stamp, weights=signal_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest