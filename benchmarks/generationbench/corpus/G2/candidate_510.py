from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Strong stocks tend to keep outperforming weak ones in the long run. "
        "This strategy aims to overweight stocks that have been strong relative to the market index, "
        "assuming a relative strength effect can be captured."
    )

    def __init__(self, lookback_period: int = 60) -> None:
        self._lookback_period = lookback_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_period)
        if closes.height < self._lookback_period:
            return Signal(information_available_at=stamp, weights={})

        nifty_closes = closes.select(pl.col("NIFTY50").alias("nifty"))
        other_closes = closes.select(pl.all().exclude("NIFTY50"))

        nifty_returns = (
            (nifty_closes["nifty"] / nifty_closes["nifty"].shift(1) - 1.0).drop_nulls()
        ).to_list()

        other_returns = {}
        for symbol in other_closes.columns:
            returns = (other_closes[symbol] / other_closes[symbol].shift(1) - 1.0).drop_nulls()
            other_returns[symbol] = returns.to_list()

        nifty_mean_return = sum(nifty_returns) / len(nifty_returns)
        other_strength = {
            symbol: max(
                [(returns[-self._lookback_period :] > nifty_mean_return).count() / self._lookback_period, 0.0]
            )
            for symbol, returns in other_returns.items()
        }

        sorted_strength = sorted(other_strength.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_strength[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest