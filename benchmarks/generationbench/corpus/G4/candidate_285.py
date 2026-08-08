from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the historical anomaly where stocks with lower volatility "
        "have tended to outperform those with higher volatility over long periods. By "
        "overweighting low-volatility stocks and underweighting high-volatility ones, we aim "
        "to capture excess returns while maintaining a balanced risk profile."
    )

    def __init__(self, window: int = 20, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        volatilities = [
            (pl.Series(closes[i : i + self._window]) / pl.Series(closes[i - 1 : i + self._window - 1]) - 1.0).std().item()
            for i in range(len(closes) - self._window)
        ]

        sorted_symbols = [symbol for symbol, _ in sorted(zip(view.symbols, volatilities), key=lambda x: x[1])]
        picks = sorted_symbols[: self._top_n]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest