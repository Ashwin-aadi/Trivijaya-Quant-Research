from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with the strongest relative performance are selected based on their "
        "highest closing prices compared to the average of all stocks in the universe. "
        "This strategy seeks to capitalize on the outperformance of leading sectors or "
        "industries within the market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_close = (history.groupby("symbol").agg(
            pl.col("adj_close").mean().alias("avg_close")
        )).with_columns(
            (pl.col("close") / pl.col("avg_close") - 1.0).alias("relative_strength")
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in avg_close.columns:
                continue
            rel_strength = float(avg_close.filter(pl.col("symbol") == symbol)["relative_strength"].item())
            if rel_strength >= max(avg_close["relative_strength"].to_list()):
                picks.append(symbol)

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest