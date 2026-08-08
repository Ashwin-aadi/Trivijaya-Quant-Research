from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies price breaks out of significant support or resistance levels and "
        "enters trades in the direction of the breakout after a confirmation period. The economic mechanism "
        "involves leveraging the tendency for prices to continue moving in the direction of the breakout "
        "after initial momentum is established, due to inertia, market sentiment, and technical analysis."
    )

    def __init__(self, lookback: int = 200, confirm_days: int = 1, stop_loss_ratio: float = 0.015) -> None:
        self._lookback = lookback
        self._confirm_days = confirm_days
        self._stop_loss_ratio = stop_loss_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Identify support and resistance levels
        supports = history.select(
            pl.col("symbol"), (pl.col("low").rolling_min(window_size=self._lookback)).alias("support")
        )
        resistances = history.select(
            pl.col("symbol"), (pl.col("high").rolling_max(window_size=self._lookback)).alias("resistance")
        )

        # Filter support and resistance levels to ensure they are distinct
        supports = (
            supports.join(resistances, on="symbol", how="outer")
            .filter(pl.col("support") < pl.col("resistance"))
            .drop_nulls(subset=["support", "resistance"])
        )

        if supports.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Detect breakouts
        breakout_candidates = []
        for symbol in view.symbols:
            if symbol not in supports.columns:
                continue

            history_for_symbol = history.filter(pl.col("symbol") == symbol)

            support, resistance = float(supports.get_symbol(symbol)["support"]), float(supports.get_symbol(symbol)["resistance"])
            high_prices = [float(v) for v in history_for_symbol["high"].to_list()[-5:]]
            low_prices = [float(v) for v in history_for_symbol["low"].to_list()[-5:]]

            if support:
                breakouts_high = any(h > resistance + 0.01 * (resistance - support) for h in high_prices)
                breakouts_low = any(l < support - 0.01 * (resistance - support) for l in low_prices)

                if breakouts_high or breakouts_low:
                    breakout_candidates.append((symbol, "high" if breakouts_high else "low"))

            elif resistance:
                breakouts_high = any(h > resistance + 0.02 * (resistance - support) for h in high_prices)
                breakouts_low = any(l < support - 0.02 * (resistance - support) for l in low_prices)

                if breakouts_high or breakouts_low:
                    breakout_candidates.append((symbol, "high" if breakouts_high else "low"))

        # Filter and rank candidates
        ranked_breakouts = sorted(breakout_candidates, key=lambda x: 100 * (resistance - support) + len(high_prices), reverse=True)

        picks = [c[0] for c in ranked_breakouts[:5]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        stop_loss_price = {
            symbol: {"high": resistance - self._stop_loss_ratio * (resistance - support), "low": support + self._stop_loss_ratio * (resistance - support)}
            for symbol in picks
        }

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
            stop_loss_prices=stop_loss_price,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest