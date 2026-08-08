from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion strategies seek to profit from the tendency of asset prices to return "
        "to their historical mean. This strategy identifies stocks that have deviated significantly"
        " from their average price over a short period and bets on them reverting back."
    )

    def __init__(self, window: int = 5, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        means = history.group_by("symbol").agg(
            (pl.col("adj_close").mean()).alias("mean")
        )
        latest_closes = pl.DataFrame({"symbol": symbols, "latest_close": view.closes(lookback=None)["session_date"]})
        merged = means.join(latest_closes, on="symbol", how="inner")

        deviations = (
            (merged["latest_close"] / merged["mean"]) - 1.0
        ).to_list()
        mean_deviation = sum(deviations) / len(deviations)
        
        weights: dict[str, float] = {}
        for idx, row in merged.iter_rows():
            symbol = row["symbol"]
            deviation = (merged.filter(pl.col("symbol") == symbol)["latest_close"] /
                         means.filter(pl.col("symbol") == symbol)["mean"]).item() - 1.0
            if abs(deviation - mean_deviation) > self._threshold:
                weight = 1.0 / len(symbols)
                weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest