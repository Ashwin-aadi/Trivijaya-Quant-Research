from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Mean reversion identifies stocks that have deviated significantly from their historical "
        "mean and are likely to revert back. By selling overvalued stocks and buying undervalued ones, "
        "we can profit from the mean-reverting behavior of stock prices."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (history["adj_close"] / history["adj_close"].shift(-1) - 1.0).mean()
        z_scores = (
            (history["adj_close"] / history["adj_close"].shift(-1) - 1.0)
            .to_list()
            .zscore(center=False, sample=True)
        )

        symbols_to_buy: list[str] = []
        symbols_to_sell: list[str] = []

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            z_score = float(z_scores.pop(0))
            if z_score > 1.5:
                symbols_to_sell.append(symbol)
            elif z_score < -1.5:
                symbols_to_buy.append(symbol)

        weight_to_buy = 0.6 / len(symbols_to_buy) if symbols_to_buy else 0
        weight_to_sell = 0.4 / len(symbols_to_sell) if symbols_to_sell else 0

        weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol in symbols_to_buy:
                weights[symbol] = weight_to_buy
            elif symbol in symbols_to_sell:
                weights[symbol] = -weight_to_sell

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest