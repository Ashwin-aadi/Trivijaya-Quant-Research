from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for past high performers to continue "
        "outperforming in the future. This phenomenon is supported by empirical evidence and can "
        "be captured by identifying symbols that have had strong returns over a recent period."
    )

    def __init__(self, window: int = 20, lookback: int = 5) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns for the recent period
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
        )
        history = history.sort("session_date", descending=True)

        # Get top performers
        top_performers = (
            history.select(pl.all().exclude("symbol", "session_date"))
                   .mean()
                   .to_dict(True)
        )

        sorted_performances = sorted(top_performers.items(), key=lambda x: float(x[1]), reverse=True)[:self._lookback]
        symbols = [s for s, _ in sorted_performances]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest