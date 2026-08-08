from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class EarningsSurpriseAnalystShift(Strategy):
    rationale = (
        "This strategy exploits the interplay between earnings surprises and analyst "
        "recommendation changes. It buys stocks that have significant earnings surprises "
        "and see a positive shift in recommendations, indicating potential price catch-up."
    )

    def __init__(self, lookback_eps: int = 4, window_recommendations: int = 5) -> None:
        self._lookback_eps = lookback_eps
        self._window_recommendations = window_recommendations

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_eps + self._window_recommendations)
        if history.height < self._lookback_eps + self._window_recommendations:
            return Signal(information_available_at=stamp, weights={})

        # Calculate earnings surprises
        eps_history = _calculate_eps_surprise(history)
        if eps_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in eps_history.columns or symbol not in history.select("symbol").columns:
                continue
            latest_eps_surprise = float(eps_history[symbol].sort(by="session_date", descending=True).head(1)[0])
            recent_recommendations = _filter_by_recommendation_changes(history, symbol)

            if (latest_eps_surprise > 0 and "Buy" in recent_recommendations) or \
               (latest_eps_surprise < 0 and "Sell" not in recent_recommendations):
                picks.append(symbol)

        picks = picks[:20]  # Limit to top 20 names
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _calculate_eps_surprise(history: pl.DataFrame) -> pl.DataFrame:
    eps_history = history.select([
        "symbol",
        ("close" - pl.col("open").shift(self._lookback_eps)).alias("eps_surprise")
    ])
    return eps_history


def _filter_by_recommendation_changes(history: pl.DataFrame, symbol: str) -> list[str]:
    recommendations = history.filter(pl.col("symbol") == symbol).select(
        "session_date", "adj_close"
    ).sort(by="session_date").tail(self._window_recommendations)
    changes = [r["adj_close"] for r in recommendations.to_dicts()]
    return ["Buy" if change > 0 else "Hold" if change == 0 else "Sell" for change in changes]


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest