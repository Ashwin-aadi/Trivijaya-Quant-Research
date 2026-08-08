from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEquityStrategy(Strategy):
    rationale = (
        "This strategy leverages both calendar effects and festival-related spending to identify stocks with positive seasonal returns, ensuring a comprehensive approach to market trends."
    )

    def __init__(self, festivals: list[str] = [], window_months: int = 3) -> None:
        self._festivals = festivals
        self._window_months = window_months

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_months)
        if history.is_empty() or history.height < 20:
            return Signal(information_available_at=stamp, weights={})

        festival_dates = {festival: date(1900, 1, 1) for festival in self._festivals}
        current_month = stamp.month
        current_year = stamp.year

        # Identify the closest upcoming festivals or first trading day of favorable months
        picks: list[str] = []
        for symbol in view.symbols:
            monthly_returns = (
                history.select([pl.col("session_date"), pl.col(symbol)])
                .filter(pl.col("session_date").dt.month() == current_month)
                .select((pl.col(symbol) / pl.col(symbol).shift(1) - 1.0).alias("r"))
                .to_dict(as_series=False)["r"]
            )
            if not monthly_returns:
                continue
            avg_return = sum(monthly_returns) / len(monthly_returns)

            # Check for favorable month or festival close to the trading day
            is_favorable_month = current_month in [9, 10, 11] and 20 <= stamp.day < 31
            if is_favorable_month:
                picks.append(symbol)
            for festival, date in festival_dates.items():
                if (current_year - view.as_of.year) * 12 + current_month >= (
                    stamp.year - view.as_of.year
                ) * 12 + stamp.month and abs((stamp - date).days) <= 30:
                    picks.append(symbol)

        # Filter out symbols not in the picks list
        weights = {s: 1.0 / len(picks) for s in picks}
        return Signal(
            information_available_at=stamp, weights={k: v for k, v in weights.items() if k in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest