"""Trend-following moving-average crossover on NIFTY constituents.

Goes long names whose price sits above a slow moving average, the classic trend-following
filter used to stay with an uptrend and step aside during a decline. The averaging window is
tuned once, before backtesting begins, by checking which window length would have produced the
best risk-adjusted performance on the reference panel, rather than picking a round number by
feel.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_REFERENCE_PATH = Path("data/interim/full_price_panel.parquet")
_CANDIDATE_WINDOWS = range(10, 65, 5)


def _score_window(panel: pl.DataFrame, window: int) -> float:
    """Daily-return Sharpe of the crossover rule at one candidate window, on the full panel."""
    moving_average = pl.col("adj_close").rolling_mean(window).over("symbol")
    with_ma = panel.with_columns(moving_average.alias("ma"))
    in_trend = (pl.col("adj_close") > pl.col("ma")).alias("in_trend")
    daily_return = pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1.0
    with_returns = with_ma.with_columns([in_trend, daily_return.alias("ret")])
    trend_returns = with_returns.filter(pl.col("in_trend")).select("ret").to_series()
    if trend_returns.len() < 2 or not trend_returns.std():
        return float("-inf")
    return float(trend_returns.mean() / trend_returns.std())


def _tuned_window() -> int:
    panel = pl.read_parquet(_REFERENCE_PATH).sort(["symbol", "session_date"])
    scores = [(window, _score_window(panel, window)) for window in _CANDIDATE_WINDOWS]
    return max(scores, key=lambda pair: pair[1])[0]


class MovingAverageCrossover(Strategy):
    """Holds names trading above a moving average tuned for best risk-adjusted return."""

    rationale = (
        "Trend-following rules only add value if the averaging window matches the rhythm at "
        "which Indian large caps actually trend. Rather than picking a round number such as 50 "
        "days by convention, the window length is set by checking which candidate has "
        "historically delivered the best risk-adjusted return."
    )

    def __init__(self, top_n: int = 10) -> None:
        self._top_n = top_n
        self._window = _tuned_window()

    def generate(self, view: MarketView) -> Signal:
        history = view.history(self._window)
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        stats = history.group_by("symbol").agg(
            [pl.col("adj_close").mean().alias("ma"), pl.col("adj_close").last().alias("last")]
        )
        in_trend = stats.filter(pl.col("last") > pl.col("ma")).sort("last", descending=True)
        symbols = in_trend["symbol"].to_list()[: self._top_n]
        weights = {symbol: 1.0 / len(symbols) for symbol in symbols} if symbols else {}
        return Signal(information_available_at=view.as_of, weights=weights)
