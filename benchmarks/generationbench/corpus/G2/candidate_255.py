from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are more likely to be overvalued due to excessive trading. "
        "By equal-weighting a screen of the least liquid stocks, we can potentially benefit "
        "from their mispricing."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = _calculate_liquidity(history)
        symbols = [symbol for symbol in view.symbols if symbol in liquidity]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        top_n = min(len(symbols), 10)  # Select top N least liquid stocks
        picks: list[str] = []
        for symbol in symbols:
            if liquidity[symbol] == float("inf"):
                continue
            picks.append(symbol)

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


def _calculate_liquidity(history: pl.DataFrame) -> dict[str, float]:
    liquidity_scores = {}
    for symbol in history["symbol"].unique().to_list():
        volume_history = (
            history.filter(pl.col("symbol") == symbol)["volume"]
            .sort("session_date")
            .to_list()
        )
        if len(volume_history) < 2:
            liquidity_scores[symbol] = float("inf")
            continue

        # Calculate daily return
        returns = [
            (float(v1) / float(v0) - 1.0)
            for v0, v1 in zip(volume_history[:-1], volume_history[1:])
        ]
        average_return = sum(returns) / len(returns)

        # Calculate liquidity score based on the absolute value of daily returns
        liquidity_scores[symbol] = abs(sum(returns)) + 2 * abs(
            min([abs(r) for r in returns])
        )
    return {k: (1 / v if v != float("inf") else v) for k, v in liquidity_scores.items()}