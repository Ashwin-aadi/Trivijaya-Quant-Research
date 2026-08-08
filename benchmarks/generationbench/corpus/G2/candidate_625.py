from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that stocks which have deviated significantly from their "
        "historical mean price will eventually revert to it. In a noisy market with many random "
        "price movements, this can provide an opportunity for profitable trading."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            (pl.col("adj_close").mean().alias("mean_close"))
        )
        recent_close = view.closes()
        
        merged = (
            history.join(recent_close, on="symbol", how="inner")
                .with_columns((pl.col("adj_close") - pl.col("mean_close")).abs().alias("deviation"))
                .select(
                    ["session_date", "symbol", "adj_close", "mean_close", "deviation"]
                )
        )

        if merged.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sorted_stocks = (
            merged.sort("deviation", descending=True)
                .head(self._threshold * self._window)
                .select(["symbol"])
        )

        symbols_to_short = [s for s in sorted_stocks["symbol"].to_list() if pl.col(s).max() > 0]
        if not symbols_to_short:
            return Signal(information_available_at=stamp, weights={})

        weight = -1.0 / len(symbols_to_short)
        return Signal(
            information_available_at=stamp, weights=dict(zip(symbols_to_short, [weight] * len(symbols_to_short)))
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest