from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy screens for stocks with higher liquidity to ensure marketability "
        "without significant price impact. Equally weighting the selected stocks aims to "
        "reduce bias towards larger-cap stocks and leverage potential mispricings in smaller, "
        "less liquid stocks."
    )

    def __init__(self, window: int = 30, threshold: float = 50.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_data = (
                history.select(["session_date", "high", "low", "volume"])
                .with_columns(
                    (pl.col("volume") / ((pl.col("high") - pl.col("low")) + 1e-6)).alias("liquidity_score")
                )
                .sort("session_date", descending=False)
                .select("liquidity_score")
            )

            if daily_data.height < self._window:
                continue
            liquidity_scores[symbol] = float(daily_data["liquidity_score"].mean().item())

        selected_symbols: list[str] = [k for k, v in liquidity_scores.items() if v >= self._threshold]
        weight = 1.0 / len(selected_symbols) if selected_symbols else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest