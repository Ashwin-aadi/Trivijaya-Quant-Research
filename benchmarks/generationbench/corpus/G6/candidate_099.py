from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy aims to maximize exposure to liquid stocks while ensuring diversification. "
        "By ranking stocks based on their liquidity over both short-term (20-day) and long-term (30-60 day) periods, "
        "we can select the top 100 most liquid stocks for equal weighting, balancing risk management with portfolio optimization."
    )

    def __init__(self, short_window: int = 20, long_window: int = 60, top_n: int = 100) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._short_window + self._long_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = _calculate_liquidity_scores(history, self._short_window, self._long_window)
        picks: list[str] = []
        for symbol in view.symbols:
            if liquidity_scores.get(symbol) is not None and liquidity_scores[symbol] >= 100:
                picks.append(symbol)

        picks = picks[: self._top_n]
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


def _calculate_liquidity_scores(history: pl.DataFrame, short_window: int, long_window: int) -> dict[str, float]:
    liquidity_scores = {}
    for symbol in view.symbols:
        if symbol not in history.columns:
            continue
        daily_volumes = [float(v) for v in history[symbol].drop_nulls().to_list()]
        if len(daily_volumes) < short_window + long_window:
            continue

        short_period_volume = sum(daily_volumes[-short_window:])
        long_period_volume = sum(daily_volumes[-long_window:])

        liquidity_score = short_period_volume / max(1, long_period_volume)
        liquidity_scores[symbol] = liquidity_score
    return {k: v for k, v in sorted(liquidity_scores.items(), key=lambda item: item[1], reverse=True)}