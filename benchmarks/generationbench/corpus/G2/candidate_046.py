from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalAdjustment(Strategy):
    rationale = (
        "Certain stocks in India may exhibit seasonal patterns due to economic activities "
        "linked to the agricultural calendar or specific holidays. By adjusting our positions "
        "based on these seasonal trends, we can exploit this phenomenon for potential returns."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or (history["session_date"].max() - history["session_date"].min()).days < 365:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        closes = history.select(["symbol", "session_date", pl.col("adj_close").alias("close")])

        seasonality_factor: dict[str, float] = {}
        for symbol in symbols:
            symbol_history = closes.filter(pl.col("symbol") == symbol).sort(by="session_date")
            mean_by_quarter = symbol_history.groupby_dynamic(
                index_column="session_date", every=f"{self._window // 4}d"
            ).agg([pl.col("close").mean().alias("mean")])

            for _, row in mean_by_quarter.iter_rows():
                seasonality_factor[row["symbol"]] = row["mean"]

        adjusted_closes: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in seasonality_factor:
                continue
            latest_close = view.latest_close()[symbol]
            adjusted_closes[symbol] = latest_close / seasonality_factor[symbol]

        top_symbols = sorted(adjusted_closes.items(), key=lambda x: x[1], reverse=True)[:5]
        weights = {s: 1.0 for s, _ in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest