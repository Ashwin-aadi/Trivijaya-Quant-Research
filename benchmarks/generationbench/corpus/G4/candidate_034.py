from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrendStrategy(Strategy):
    rationale = (
        "The Indian equity market exhibits significant seasonal trends influenced by cultural "
        "festivals like Diwali and government policies around election periods. By identifying "
        "stocks with historical outperformance during these events, we can exploit these "
        "patterns to generate profitable trading signals."
    )

    def __init__(self, window: int = 365, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_scores = _compute_seasonality_scores(history)
        sorted_symbols = [symbol for symbol, score in seasonality_scores.items()]
        picks: list[str] = sorted_symbols[: self._top_n]
        
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_seasonality_scores(history: pl.DataFrame) -> dict[str, float]:
    seasonality_factors = {
        "Diwali": [date(2023, 10, 27), date(2024, 10, 26)],
        # Add more seasonal events as needed
    }
    
    scores = {}
    for symbol in history["symbol"].unique().to_list():
        symbol_data = history.filter(pl.col("symbol") == symbol)
        recent_closes = symbol_data.select(
            pl.col("session_date").filter((pl.col("session_date") >= seasonality_factors["Diwali"][0]) & (pl.col("session_date") <= seasonality_factors["Diwali"][1])),
            pl.col("close")
        ).sort("session_date")

        if recent_closes.height < 2:
            continue

        last_close = float(recent_closes.select(pl.last("close")).to_series().item())
        avg_close = float(recent_closes.select(pl.mean("close")).to_series().item())

        # Simple return-based score
        score = (last_close - avg_close) / avg_close
        scores[symbol] = score

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_scores