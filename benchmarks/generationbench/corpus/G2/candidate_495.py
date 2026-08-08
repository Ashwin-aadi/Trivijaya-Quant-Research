from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Highly liquid stocks are often those that are most actively traded and thus have "
        "lower bid-ask spreads. Investors may be willing to pay a premium for liquidity, "
        "suggesting that highly liquid stocks can outperform less liquid ones."
    )

    def __init__(self, min_trading_days: int = 10) -> None:
        self._min_trading_days = min_trading_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._min_trading_days)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores: dict[str, float] = {}
        for symbol in view.symbols:
            trading_days = (
                history.select(pl.col("session_date").unique().shape[0])
                .to_series()
                .filter(pl.col(0) == symbol)
                .sum()
                .to_list()[0]
            )
            if trading_days >= self._min_trading_days:
                liquidity_scores[symbol] = 1.0 / trading_days

        if not liquidity_scores:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_scores)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquidity_scores},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest