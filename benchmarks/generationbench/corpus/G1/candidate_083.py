from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentumAndVolatility(Strategy):
    rationale = (
        "This strategy combines momentum and volatility to identify stocks with strong"
        " recent performance but low future expected volatility. The idea is that such"
        " stocks may continue their upward trend."
    )

    def __init__(self, window1: int = 20, window2: int = 60) -> None:
        self._window1 = window1
        self._window2 = window2

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window1, self._window2))
        if history.height < max(self._window1, self._window2):
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            close_prices = [float(v) for v in history.select(pl.col(symbol)).drop_nulls().to_list()[0]]
            momentum_score = (close_prices[-1] - close_prices[0]) / sum(close_prices)

            volatility_score = pl.DataFrame(
                {"close": close_prices}
            ).with_columns((pl.col("close").rolling_std(window=self._window2) / 2).alias("volatility")).select(
                pl.col("volatility")
            ).row(0)[0]

            momentum_scores[symbol] = momentum_score
            volatility_scores[symbol] = volatility_score

        sorted_symbols = [
            s for _, s in sorted(momentum_scores.items(), key=lambda item: (item[1], -volatility_scores[item[0]]), reverse=True)
        ]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols[:5]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest