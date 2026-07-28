"""Honest long-run value strategy that happens to use the words 'full' and 'entire'.

Negative-control fixture: ``entire_range`` and ``full_window`` both name the same thing --
every session ``view`` has revealed so far. Because ``MarketView`` truncates at construction
time to sessions strictly before the decision date, "the entire visible range" and "a full-sample
statistic computed over train and test together" are not the same thing: this strategy's mean is
taken only over history that already existed at the moment of the decision.
"""

from __future__ import annotations

from datetime import date

from src.backtest.strategy import MarketView, Signal, Strategy


def _latest_visible_session(view: MarketView) -> date:
    """The newest session already printed to the tape, the only date a signal may cite."""
    recent = view.history(1)
    if recent.is_empty():
        return date.min
    latest = recent["session_date"].max()
    assert isinstance(latest, date)
    return latest


class TrapFullAndEntire(Strategy):
    """Buy names trading at a discount to their entire visible-history average."""

    rationale = (
        "A price well below the average level it has traded at over its whole visible history "
        "is a simple long-run mean-reversion signal: absent new information, prices are assumed "
        "to drift back toward levels they have historically occupied. Requiring a minimum number "
        "of visible sessions before scoring a name keeps the estimate from being driven by a "
        "handful of noisy early observations."
    )

    def __init__(self, min_sessions: int = 100, top_n: int = 5) -> None:
        self.min_sessions = min_sessions
        self.top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        decision_date = _latest_visible_session(view)
        # "entire_range" -- every session view has revealed, unlimited by any lookback. Still
        # bounded above by as_of because MarketView truncated it before this method ever ran.
        entire_range = view.closes()
        if entire_range.height < self.min_sessions:
            return Signal(information_available_at=decision_date, weights={})
        full_window = entire_range  # same frame, named again to make the equivalence explicit
        latest_row = full_window.tail(1).row(0, named=True)
        discount: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in full_window.columns:
                continue
            series = full_window[symbol].drop_nulls()
            if series.len() < self.min_sessions:
                continue
            long_run_mean = series.mean()
            current = latest_row.get(symbol)
            if current is None or long_run_mean is None or current >= long_run_mean:
                continue
            discount[symbol] = (long_run_mean - current) / long_run_mean
        ranked = sorted(discount.items(), key=lambda kv: kv[1], reverse=True)[: self.top_n]
        weights: dict[str, float] = {}
        if ranked:
            share = 1.0 / len(ranked)
            for symbol, _gap in ranked:
                weights[symbol] = share
        return Signal(information_available_at=decision_date, weights=weights)
