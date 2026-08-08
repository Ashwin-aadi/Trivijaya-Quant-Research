from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum strategy exploits the tendency for stocks that have "
        "performed well in the recent past to continue performing well. This strategy allocates "
        "weights based on the historical performance of each stock."
    )

    def __init__(self, window: int = 60, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window or closes.width <= 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            daily_returns = (closes[symbol].drop_nulls().to_list()[self._window - 1:] / 
                             [float(closes[symbol].shift(1).drop_nulls().to_list()[i]) for i in range(self._window - 1, len(closes[symbol]))] - 1.0)
            mean_return = sum(daily_returns) / len(daily_returns)
            momentum_scores.append((symbol, mean_return))

        sorted_momentum_scores = sorted(momentum_scores, key=lambda x: x[1], reverse=True)[:self._top_n]
        picks = [score[0] for score in sorted_momentum_scores]

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