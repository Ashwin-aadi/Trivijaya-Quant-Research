from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "Historical data often shows that certain stocks exhibit higher returns during specific "
        "months of the year. This strategy exploits such seasonality effects by overweightsing "
        "the stocks with better performance in those months."
    )

    def __init__(self, lookback_window: int = 10, min_monthly_trades: int = 5) -> None:
        self._lookback_window = lookback_window
        self._min_monthly_trades = min_monthly_trades

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_window)
        if closes.height < self._lookback_window * 21:
            return Signal(information_available_at=stamp, weights={})

        seasonal_performance: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            monthly_closes = _monthly_returns(closes, symbol)
            valid_months = [m for m, r in monthly_closes.items() if abs(r) > 0.01]
            if len(valid_months) < self._min_monthly_trades:
                continue
            avg_return = sum(monthly_closes[m] for m in valid_months) / len(valid_months)
            seasonal_performance[symbol] = avg_return

        top_performers = sorted(seasonal_performance.items(), key=lambda x: x[1], reverse=True)
        picks, _ = zip(*top_performers[:5])
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
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


def _monthly_returns(closes: pl.DataFrame, symbol: str) -> dict[int, float]:
    monthly_closes = {}
    for month in range(1, 13):
        start_date = f"{2000 + (month - 1) // 12}-{month % 12 + 1}-01"
        end_date = (
            f"{2000 + (month - 1) // 12}-{month % 12 + 1}-28" if month != 2 else "2000-02-29"
        )
        monthly_closes[month] = float(closes
                                      .filter(pl.col("session_date").gte(start_date)
                                             .lt(end_date))
                                      [symbol]
                                      .mean())
    return monthly_closes