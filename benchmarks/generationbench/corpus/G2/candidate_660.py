from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum identifies stocks that have outperformed their peers over a "
        "recent period. Such stocks are expected to continue outperforming in the near future, "
        "based on the notion that past performance is indicative of future returns."
    )

    def __init__(self, window: int = 20, lookback: int = 5) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            symbol_history = closes[[symbol]].sort("session_date").to_pandas()
            if len(symbol_history) < self._window + 1:
                continue

            # Calculate the daily returns over the lookback period.
            daily_returns = (
                (symbol_history["adj_close"].iloc[-self._lookback:] / 
                 symbol_history["adj_close"].iloc[-(self._lookback + 1)] - 1.0)
            ).to_list()

            # Compute the mean return over the lookback period.
            mean_return = sum(daily_returns) / len(daily_returns)

            momentum_scores[symbol] = mean_return

        sorted_scores = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_scores[: self._lookback]]

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
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest