from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Indian markets exhibit predictable seasonal effects where specific months or days "
        "show higher returns. By identifying these patterns and exploiting them through a "
        "rule-based strategy, we aim to capture excess returns."
    )

    def __init__(self, window: int = 500, favorable_months: list[int] = [12]) -> None:
        self._window = window
        self._favorable_months = favorable_months

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            monthly_returns = (
                history.filter(pl.col("session_date").dt.month().is_in(self._favorable_months))
                .select(
                    pl.col("symbol"),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
                )
            ).group_by("symbol").agg(pl.col("r").mean().alias("avg_return"))
            if not monthly_returns.height:
                continue
            avg_returns[symbol] = float(monthly_returns.select("avg_return")[0])

        top_symbols = sorted(avg_returns.keys(), key=lambda s: avg_returns[s], reverse=True)[:30]
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