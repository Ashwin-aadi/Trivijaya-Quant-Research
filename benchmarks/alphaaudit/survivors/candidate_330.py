from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength (i.e., outperforming the market) are expected to "
        "outperform in the future. By allocating more weight to these stocks, we can capture "
        "positive momentum."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_close = view.latest_close()[view.symbols[0]]
        strengths: list[float] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            market_adjusted_close = [v / market_close * 100.0 for v in values]
            strength = (market_adjusted_close[-1] - min(market_adjusted_close)) / (
                max(market_adjusted_close) - min(market_adjusted_close)
            )
            strengths.append(strength)

        top_strengths = sorted(zip(view.symbols, strengths), key=lambda x: x[1], reverse=True)[: self._top_n]
        picks = [tup[0] for tup in top_strengths]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest