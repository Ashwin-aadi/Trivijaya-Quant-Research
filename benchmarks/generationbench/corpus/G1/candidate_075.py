from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a key indicator of market efficiency. Higher liquidity often "
        "corresponds to better price discovery and lower transaction costs. This strategy "
        "equal-weights the top N most liquid stocks based on their volume, aiming for "
        "a balanced portfolio across highly traded securities."
    )

    def __init__(self, top_n: int = 10) -> None:
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)  # Using a lookback of 365 days for volume
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            volume = [float(v) for v in history[f"{symbol}_volume"].to_list()]
            if len(volume) < 365:
                continue
            daily_avg_volume = sum(volume) / 365.0

            if daily_avg_volume > 1_000_000:  # Assuming a threshold of 1 million for liquidity
                liquidity_screened.append(symbol)

        liquidity_screened = liquidity_screened[: self._top_n]
        if not liquidity_screened:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_screened)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquidity_screened},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest