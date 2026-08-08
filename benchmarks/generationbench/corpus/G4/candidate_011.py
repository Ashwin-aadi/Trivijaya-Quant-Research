from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumEarningsComposite(Strategy):
    rationale = (
        "This strategy combines market momentum and earnings surprise to identify "
        "stocks with strong momentum but unexpected financial performance. By "
        "leveraging these two weakly related characteristics, we aim to capitalize on "
        "both price history trends and actual company performance deviations."
    )

    def __init__(self, window: int = 52, earnings_lookback_quarters: int = 1) -> None:
        self._window = window
        self._earnings_lookback_quarters = earnings_lookback_quarters

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = self._calculate_momentum_scores(history)
        earnings_surprise_scores = self._calculate_earnings_surprise_scores(
            view, stamp
        )

        composite_scores = (
            momentum_scores.to_list()
            + [self._compute_composite_score(mom, eps) for mom, eps in zip(
                momentum_scores.drop_nulls().to_list(), earnings_surprise_scores.drop_nulls().to_list())
            ]
        )
        if not composite_scores:
            return Signal(information_available_at=stamp, weights={})

        top_decile = int(len(view.symbols) * 0.1)
        selected_symbols = sorted(
            zip(view.symbols, composite_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_decile]

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in selected_symbols}
        )

    def _calculate_momentum_scores(self, history: pl.DataFrame) -> pl.Series:
        symbols = view.symbols
        momentum_scores = [
            (history.select(pl.col(s).tail(1)[0] / history.select(pl.col(s).head(1))[0] - 1.0)).item()
            for s in symbols
        ]
        return pl.Series(name="momentum_score", values=momentum_scores)

    def _calculate_earnings_surprise_scores(self, view: MarketView, stamp: date) -> pl.Series:
        earnings_history = view.history(lookback=self._earnings_lookback_quarters)
        earnings_scores = [
            (float(eps) if not eps.is_nan() else 0.0)
            for s in symbols
            for eps in earnings_history.select(pl.col(s)).tail(1)[0]
        ]
        return pl.Series(name="earnings_score", values=earnings_scores)

    def _compute_composite_score(self, momentum: float, earnings: float) -> float:
        return 0.5 * momentum + 0.5 * earnings

def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest