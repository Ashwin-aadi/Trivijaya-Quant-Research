from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of financial assets to return to their "
        "mean price over time. Short-horizon mean reversion looks for extreme deviations from this "
        "mean and bets on a return towards it."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        means = history.groupby("symbol").agg(
            (pl.col("adj_close").mean().alias("mean"))
        )
        latest_prices = view.closes()
        deviations = (
            latest_prices.join(means.select(["symbol", "mean"]), on="symbol")
            .with_columns((pl.col("adj_close") - pl.col("mean")).alias("deviation"))
            .filter(pl.col("deviation").is_not_null())
            .sort("deviation")
        )

        if deviations.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in deviations.to_dicts()[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest