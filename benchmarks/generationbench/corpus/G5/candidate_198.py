from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "A stock with a higher relative strength compared to the broad market "
        "tends to outperform over the long run. This strategy identifies stocks "
        "that have performed better than their peers in the recent past."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean of NIFTY 100 index
        broad_market_mean = (
            history.select(pl.col("NIFTY_100").mean().alias("broad_market"))
        )

        # Compute relative strength for each stock in the universe
        individual_strengths = (
            history.select([pl.col(symbol).mean().alias(symbol)
                            for symbol in view.symbols])
            .with_column(broad_market_mean["broad_market"])
        )

        relative_strengths = (individual_strengths[view.symbols] / broad_market_mean["broad_market"] - 1.0).to_dicts()

        # Sort symbols by their latest relative strength
        top_n_strongest: list[str] = sorted(
            view.symbols,
            key=lambda symbol: relative_strengths[symbol][-1],
            reverse=True
        )[:5]

        if not top_n_strongest:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_strongest)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_strongest}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest