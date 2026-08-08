from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighing(Strategy):
    rationale = (
        "Liquidity is a proxy for market confidence and tradability. By focusing on the most liquid "
        "stocks, we can ensure that our trades are less likely to impact the price of these stocks, "
        "potentially leading to more stable returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores: list[float] = []
        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        volumes = [
            float(v) for v in history[symbols].select(pl.col("volume").sum()).to_series().to_list()[0]
        ]
        total_volume = sum(volumes)

        for volume in volumes:
            liquidity_scores.append(volume / total_volume)

        top_n_symbols = sorted(zip(symbols, liquidity_scores), key=lambda x: x[1], reverse=True)[
                        : self._top_n
                    ]
        weights = {symbol: 1.0 / len(top_n_symbols) for symbol, _ in top_n_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    return newest