from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategy exploits the tendency for prices to continue moving "
        "beyond initial resistance or support levels due to market momentum and trader behavior. "
        "By identifying key levels and confirming breakouts with high volume, we can capitalize on "
        "continued price movement."
    )

    def __init__(self, window: int = 90, min_volume_factor: float = 2.0) -> None:
        self._window = window
        self._min_volume_factor = min_volume_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_candidates: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            data = history.select(
                pl.col("session_date"), pl.col(symbol).alias("close"), pl.col(symbol + "_volume").alias("volume")
            )

            # Calculate daily range and breakout condition
            high_low_range = (data["high"] - data["low"]).alias("range")
            data = data.with_columns(high_low_range)
            breakout_condition = (
                (data.select(pl.last("close")).item() > pl.col("high").max()) |
                (data.select(pl.last("close")).item() < pl.col("low").min())
            )
            
            # Check breakout condition
            if breakout_condition.any():
                last_close_price = data["close"].last().item()
                breakout_date = data.filter(breakout_condition).select(pl.first("session_date")).item().date()

                # Ensure volume on breakout day is high
                min_volume = history.filter(pl.col("symbol") == symbol)["volume"].mean().item() * self._min_volume_factor
                if (history.filter(
                    (pl.col("symbol") == symbol) & 
                    (pl.col("session_date") == breakout_date)
                )["volume"].sum().item() > min_volume):
                    direction = "bullish" if last_close_price >= history.select(pl.last("high")).item() else "bearish"
                    days_above_resistance = data.filter(
                        pl.col("close") > (history.select(pl.last("high")).item() if direction == "bullish" else history.select(pl.last("low")).item())
                    ).shape[0]
                    
                    breakout_candidates[symbol] = (
                        float(breakout_date) + days_above_resistance * 1
                    )  # Score based on breakout date and persistence

        # Rank candidates by their score
        ranked_symbols = sorted(breakout_candidates.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in ranked_symbols[:20]]  # Top 20 symbols

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
    newest = visible["session_date"].max().date()
    assert isinstance(newest, date)
    return newest