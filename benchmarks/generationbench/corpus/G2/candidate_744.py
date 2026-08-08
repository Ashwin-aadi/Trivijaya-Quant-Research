from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that stocks with strong relative performance "
        "in the recent past are likely to continue outperforming. This strategy exploits "
        "the tendency of high-momentum stocks to maintain their relative strength."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "session_date" not in history.columns:
                continue

            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            dates = [date.fromisoformat(date_str) for date_str in history["session_date"].to_list()]

            close_series = pl.Series(closes)
            momentum_score = (close_series - close_series.shift(self._window)).rank(
                method="ordinal", descending=True
            ).to_list()[0]

            if momentum_score == 1:
                momentum_scores[symbol] = dates[-1]

        # Select the top N symbols based on their momentum scores
        top_symbols = [k for k, v in sorted(momentum_scores.items(), key=lambda item: -item[1])]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest