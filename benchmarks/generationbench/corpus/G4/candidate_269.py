from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies stocks that have recently broken out of consolidation "
        "patterns and then wait for a retracement before entering the trade. It leverages "
        "momentum and trend continuation to identify higher probability trades."
    )

    def __init__(self, window: int = 50, lookback: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback + 10)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        consolidation_ranges = _calculate_consolidation_ranges(history[symbols])
        breakout_candidates = _find_breakouts(consolidation_ranges, history)
        ranked_candidates = _rank_candidates(breakout_candidates)

        picks: list[str] = []
        for symbol in symbols:
            if symbol not in ranked_candidates or len(ranked_candidates[symbol]) < self._top_n:
                continue
            top_breakout = ranked_candidates[symbol][-1]
            weight = 1.0 / min(self._top_n, len(ranked_candidates[symbol]))
            picks.append((symbol, weight))

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights=dict(picks),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_consolidation_ranges(df: pl.DataFrame) -> dict[str, tuple[float, float]]:
    ranges = {}
    for symbol in df.columns[1:]:
        highs = df[symbol].to_list()[1:]
        lows = [0.0] + df[f"{symbol}_low"].to_list()[:-1]
        range_highs = [max(h, l) for h, l in zip(highs, lows)]
        range_lows = [min(h, l) for h, l in zip(highs, lows)]
        ranges[symbol] = (min(range_lows), max(range_highs))
    return ranges


def _find_breakouts(ranges: dict[str, tuple[float, float]], history: pl.DataFrame) -> dict[str, list[tuple[date, bool]]]:
    breakouts = {}
    for symbol in ranges:
        if symbol not in history.columns:
            continue
        closes = [float(v) for v in history[symbol].to_list()]
        high_range, low_range = ranges[symbol]
        breakout_events = []
        for i, close in enumerate(closes):
            if i < len(closes) - 1 and (close > high_range or close < low_range):
                breakout = close > high_range
                breakout_events.append((history["session_date"][i], breakout))
        breakouts[symbol] = breakout_events
    return breakouts


def _rank_candidates(breakouts: dict[str, list[tuple[date, bool]]]) -> dict[str, list[str]]:
    ranked = {}
    for symbol in breakouts:
        events = [event[0] for event in breakouts[symbol]]
        ranked[symbol] = sorted(events, key=lambda x: -events.count(x))
    return ranked