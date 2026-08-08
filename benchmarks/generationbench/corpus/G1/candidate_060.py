from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well relative to the market in recent periods to continue outperforming. This "
        "strategy ranks assets based on their performance and invests in the top performers."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

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

            # Calculate the momentum score as the percentage change from the first to last close
            momentum_score = (values[-1] - values[0]) / abs(values[0])
            momentum_scores[symbol] = momentum_score

        top_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[: self._top_n]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        total_weight = 1.0
        weight_per_symbol = total_weight / len(top_symbols)
        weights = {symbol: weight_per_symbol for symbol, _ in top_symbols}
        
        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest