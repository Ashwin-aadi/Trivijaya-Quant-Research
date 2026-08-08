from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "This strategy exploits historical seasonality and calendar effects in the Indian market. "
        "It identifies months with higher returns for specific sectors and makes trades 1-2 weeks before those periods."
    )

    def __init__(self, window: int = 5 * 365 // 12, favorable_months: list[int] = [8]) -> None:
        self._window = window
        self._favorable_months = favorable_months

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history["session_date"].max() < date(year=stamp.year - 1, month=1, day=1):
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        favorable_symbols: dict[str, float] = {}

        for symbol in symbols:
            if symbol not in history.columns:
                continue

            monthly_data = (
                history.filter(pl.col("session_date").dt.month().is_in(self._favorable_months))
                .select(
                    pl.col("session_date").dt.year(),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
                )
            )

            if monthly_data.height < len(self._favorable_months):
                continue

            avg_return = monthly_data.select(pl.col("return").mean()).item()
            favorable_symbols[symbol] = avg_return

        # Rank symbols based on their average return
        ranked_symbols = sorted(favorable_symbols.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in ranked_symbols[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest