from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumBasedStrategy(Strategy):
    rationale = (
        "This strategy leverages the cross-sectional momentum effect by identifying "
        "high-performing stocks over a recent period and allocating higher weights to "
        "them in the portfolio. It aims to capture excess returns from past winners' "
        "tendency to continue outperforming."
    )

    def __init__(self, lookback: int = 120, top_n: int = 30) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty() or history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            opens = [float(v) for v in history[symbol]["open"].to_list()]
            closes = [float(v) for v in history[symbol]["close"].to_list()]
            if len(opens) < self._lookback or len(closes) < self._lookback:
                continue

            cumulative_return = (closes[-1] - opens[0]) / opens[0]
            momentum_scores[symbol] = cumulative_return

        sorted_symbols = [
            symbol
            for _, symbol in sorted(
                momentum_scores.items(), key=lambda item: -item[1]
            )
        ][: self._top_n]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest