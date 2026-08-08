from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies and exploits breakout continuation patterns. "
        "Breakouts from significant support or resistance levels are identified, confirmed by price action, and followed by a target based on projected trends."
    )

    def __init__(self, lookback: int = 50, top_n: int = 10) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        candidates = []
        for symbol in view.symbols:
            if symbol not in history.symbol.unique():
                continue

            hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
            opens = hist["open"].to_list()
            closes = hist["close"].to_list()

            # Identify potential breakouts
            breakout_price = max(closes[-20:]) if len(closes) > 20 else max(closes)
            low, high = min(opens), max(opens)

            for i in range(self._lookback - 1):
                if (i >= self._lookback // 2 and
                        closes[i] <= breakout_price < opens[i] or
                        closes[i] > high * 0.95 and high > low * 1.05):
                    continue

                # Confirm the breakout with a confirmation bar
                confirm = i + 1 < self._lookback and (
                    (closes[i] <= open_ for open_ in opens[:i+1]) and
                    closes[i+1] > max(opens)
                ) or (
                    (high * 0.95 >= close for close in closes[:i+1]) and
                    high < min(opens)
                )
                if confirm:
                    candidates.append(symbol)

        # Rank candidates based on breakout strength, duration, and volume
        ranked = _rank_candidates(candidates, history)
        picks = ranked[: self._top_n]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _rank_candidates(candidates: list[str], history: pl.DataFrame) -> list[str]:
    ranked = []
    for symbol in candidates:
        hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
        last_price = float(hist.select(pl.last("adj_close"))[0, 0])
        strength = abs(last_price - max(hist["high"][:20]))
        duration = len(hist) - 1
        volume = hist["volume"].sum()
        rank_score = (strength + duration * 0.5 + volume / 1e6) / 3.0
        ranked.append((symbol, rank_score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in ranked]