from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion occurs when asset prices return to the historical mean over a given "
        "time period. Short-horizon mean reversion strategies aim to capitalize on price "
        "deviations from this mean by betting that the price will revert back."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = history.to_vertical().select(
            pl.col("symbol"), pl.col("adj_close").alias(f"close_{self._window}")
        )

        means = (
            symbol_prices.groupby("symbol")
            .agg(pl.col(f"close_{self._window}").mean().alias("mean"))
            .collect()
        )
        latest_closes = view.closes(lookback=self._window).select(
            pl.col("symbol"), pl.col("adj_close").alias("latest_close")
        )

        merged = means.join(latest_closes, on="symbol", how="inner")
        relative_distances = (
            (merged.latest_close - merged.mean) / merged.mean
        ).abs().to_series()

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in relative_distances.keys():
                continue
            distance = relative_distances.get(symbol)
            if distance > 0.25 and (merged.filter(pl.col("symbol") == symbol).mean <
                                    merged.latest_close[symbol]):
                picks.append(symbol)

        weight = 1.0 / len(picks) if picks else 0.0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest