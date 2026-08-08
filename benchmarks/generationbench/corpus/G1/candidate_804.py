from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum seeks to identify stocks that have outperformed their peers "
        "over a recent period. These stocks are expected to continue outperforming in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            returns = [(values[i + 1] - values[i]) / values[i] for i in range(len(values[:-1]))]
            score = sum(returns[-5:]) / max(len(returns), 1)
            momentum_scores[symbol] = score

        sorted_symbols = [k for k, v in sorted(momentum_scores.items(), key=lambda item: -item[1])]
        top_n_symbols = sorted_symbols[:3]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest