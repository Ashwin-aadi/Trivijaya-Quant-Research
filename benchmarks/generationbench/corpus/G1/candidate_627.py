from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity screening helps identify stocks that are easier to trade without affecting "
        "the market price. Equal weighting across these liquid stocks can provide a balanced and"
        "trading-friendly portfolio."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        liquidity_scores = _calculate_liquidity_scores(history[symbols])
        total_score = sum(liquidity_scores.values())

        if not symbols or total_score == 0:
            return Signal(information_available_at=stamp, weights={})

        weights = {symbol: score / total_score for symbol, score in liquidity_scores.items()}
        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items() if w > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_liquidity_scores(df: pl.DataFrame) -> dict[str, float]:
    liquidity_scores = {}
    for symbol in df.columns[:-3]:  # Exclude 'symbol', 'session_date', and 'volume'
        daily_changes = (df[symbol] / df[symbol].shift(1) - 1.0).drop_nulls().to_list()
        mean_change = sum(daily_changes) / len(daily_changes)
        volume = df["volume"].sum().item()
        score = abs(mean_change) * volume
        liquidity_scores[symbol] = score
    return liquidity_scores