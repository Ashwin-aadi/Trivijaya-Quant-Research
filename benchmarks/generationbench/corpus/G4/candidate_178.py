from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityBasedStrategy(Strategy):
    rationale = (
        "Seasonal patterns in Indian equity markets suggest that certain sectors or stocks exhibit "
        "unusually high returns during specific months. This strategy leverages historical data to identify "
        "these patterns and exploit them for profit."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        historical_returns = self._calculate_historical_returns(closes, symbols)

        top_n_symbols = sorted(historical_returns.items(), key=lambda x: -x[1])[: self._top_n]
        picks = [symbol for symbol, _ in top_n_symbols]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )

    def _calculate_historical_returns(self, closes: pl.DataFrame, symbols: tuple[str, ...]) -> dict[str, float]:
        returns_dict = {}
        for symbol in symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            mean_return = sum(v - values[0] for v in values[-self._window:]) / (self._window - 1)
            returns_dict[symbol] = mean_return

        return returns_dict


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest