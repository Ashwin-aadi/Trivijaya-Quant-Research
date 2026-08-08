from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrend(Strategy):
    rationale = (
        "This strategy exploits the tendency of markets to revert to historical trends after "
        "significant volatility events. It identifies trending assets based on past performance and scales entry by current market volatility."
    )

    def __init__(self, trend_window: int = 50, volatility_window: int = 20, top_n: int = 30) -> None:
        self._trend_window = trend_window
        self._volatility_window = volatility_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._volatility_window)
        if closes.height < self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        trend_scores = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            returns = [(close_series[i] - close_series[i-1]) / close_series[i-1] for i in range(1, len(close_series))]
            volatility = pl.DataFrame({"returns": returns}).select(pl.col("returns").std()).to_series().item()
            slope = _linear_slope(close_series[-self._trend_window:])
            trend_score = slope / volatility if volatility > 0 else 0
            trend_scores[symbol] = trend_score

        ranked_symbols = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        weights = {s: 5.0 / len(ranked_symbols) for s, _ in ranked_symbols}
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _linear_slope(series: list[float]) -> float:
    days = range(len(series))
    slope, intercept = pl.DataFrame({"days": days, "close": series}).select(
        (pl.col("days") * pl.col("close")).sum() - (pl.col("days").sum() * pl.col("close").sum()) / len(days),
        (len(days) * pl.col("close").mean() - pl.col("days").sum() * pl.col("close").mean())
    ).to_series().item(), 0
    return slope / (len(series) * sum((day - series[0])**2 for day in days))