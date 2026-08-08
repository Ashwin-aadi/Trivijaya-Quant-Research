from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies and capitalizes on breakout continuation patterns in the Indian market. "
        "It leverages the tendency for prices to continue moving after initial strong momentum, "
        "potentially benefiting from established trends post-breakout."
    )

    def __init__(self, window: int = 50, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_candidates: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            opens = [float(v) for v in df["open"].to_list()]
            closes = [float(v) for v in df["close"].to_list()]
            highs = [float(v) for v in df["high"].to_list()]
            lows = [float(v) for v in df["low"].to_list()]
            volumes = [float(v) for v in df["volume"].to_list()]

            support, resistance = _find_support_resistance(opens, closes)
            recent_close = view.latest_close()[symbol]

            breakout_price = max(resistance[-1], opens[0])
            if recent_close > breakout_price:
                continue

            confirmed_breakout = False
            for i in range(1, len(closes)):
                if closes[i] >= resistance[-1]:
                    confirmed_breakout = True
                    break

            if not confirmed_breakout:
                continue

            upper_bound = min(resistance[-1], recent_close * 1.05)
            lower_bound = max(support[-1], recent_close * 0.95)

            in_range = all(lower_bound <= v < upper_bound for v in closes[-self._window:])
            if not in_range:
                continue

            volume_ratio = volumes[-1] / sum(volumes[-self._window:]) * 100
            magnitude = abs(resistance[-1] - support[-1])

            score = (magnitude + volume_ratio) / 2.0

            breakout_candidates.append((symbol, score))

        breakout_candidates.sort(key=lambda x: x[1], reverse=True)
        picks = [candidate[0] for candidate in breakout_candidates[: self._top_n]]
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


def _find_support_resistance(opens: list[float], closes: list[float]) -> tuple[list[float], list[float]]:
    high_points = [high for high in opens + closes if high == max(opens + closes)]
    low_points = [low for low in opens + closes if low == min(opens + closes)]

    resistance_levels = sorted(set(high_points), reverse=True)[:5]
    support_levels = sorted(set(low_points))[:5]

    return support_levels, resistance_levels