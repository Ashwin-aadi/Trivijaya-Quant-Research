from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for assets that have performed "
        "well in the recent past to continue outperforming. This phenomenon can be attributed "
        "to various factors such as market inefficiencies and herding behavior."
    )

    def __init__(self, lookback_window: int = 60, top_n: int = 10) -> None:
        self._lookback_window = lookback_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_window)
        if closes.height < self._lookback_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_series) < self._lookback_window:
                continue

            returns = [(close_series[i] / close_series[i - 1] - 1.0) for i in range(1, self._lookback_window)]
            momentum_score = sum(returns) / len(returns)
            momentum_scores.append(momentum_score)

        top_symbols = [symbol for _, symbol in sorted(zip(momentum_scores, view.symbols), reverse=True)[:self._top_n]]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest