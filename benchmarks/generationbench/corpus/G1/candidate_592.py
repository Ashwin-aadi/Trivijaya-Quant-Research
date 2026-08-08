from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceLevelReversion(Strategy):
    rationale = (
        "Price reversion is a common market phenomenon where prices tend to move back towards "
        "the mean after extreme deviations. This strategy aims to capture such movements by buying "
        "undervalued stocks and selling overvalued ones based on their recent price levels."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().to_dict(True)["adj_close"]
        z_scores = (
            closes.with_columns(
                (pl.col("adj_close") - pl.lit(mean_close)) / pl.col("adj_close").std().alias("z_score")
            )
        ).select("symbol", "session_date", "z_score")

        signals: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in z_scores.columns:
                continue
            latest_z_score = z_scores.filter(pl.col("session_date") == stamp).get("z_score").to_list()[0]
            if abs(latest_z_score) > 1.5:  # Threshold for reversion signal
                signals.append((symbol, -latest_z_score / 2))  # Adjust weights based on z-score

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(abs(weight) for _, weight in signals)
        adjusted_weights = {s: w / total_weight for s, w in signals}
        cash_allocation = 1.0 - sum(adjusted_weights.values())
        adjusted_weights["CASH"] = cash_allocation

        return Signal(information_available_at=stamp, weights=adjusted_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest