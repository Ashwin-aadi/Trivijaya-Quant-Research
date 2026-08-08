from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Stocks in India may exhibit seasonal trends due to the impact of monsoons and other "
        "seasonal factors on agricultural and related sectors. Identifying these trends can provide"
        "opportunities for profitable trading."
    )

    def __init__(self, lookback_years: int = 3, threshold: float = 0.1) -> None:
        self._lookback_years = lookback_years
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_year = (view.as_of.year - self._lookback_years + 1) % 4
        symbols = [s for s in view.symbols if s.startswith("NIFTY")]
        
        seasonals: dict[str, float] = {}
        for symbol in symbols:
            year_data = history.filter(pl.col("session_date").dt.year() == latest_year)
            if year_data.height < 252:
                continue

            closes = year_data.select(pl.col("adj_close"))
            mean_close = closes.mean().item()
            seasonal_strength = (year_data.sort("session_date").tail(20)["adj_close"].mean().item() - mean_close) / mean_close
            seasonals[symbol] = seasonal_strength if abs(seasonal_strength) > self._threshold else 0.0

        sorted_seasonals = sorted(seasonals.items(), key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in sorted_seasonals[:5]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={p: weight for p in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest