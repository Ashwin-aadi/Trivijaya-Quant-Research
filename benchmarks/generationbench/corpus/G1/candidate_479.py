from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two simple characteristics: short-term momentum and "
        "long-term support. Short-term momentum signals strength over the past 10 days, while "
        "support levels indicate areas where prices have historically found a floor."
    )

    def __init__(self, short_window: int = 10, long_window: int = 50) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._long_window + 1)
        if closes.height < self._long_window + 1:
            return Signal(information_available_at=stamp, weights={})

        short_momentum = pl.Series([float(v) for v in closes["close"].to_list()[-self._short_window:]])
        long_support = _find_long_term_support(closes)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            short_momentum_factor = (closes[symbol][-1] - short_momentum.mean()) / short_momentum.std()
            support_factor = 1.0 if closes[symbol][-1] >= long_support[symbol] else 0.0

            if short_momentum_factor > 0.5 and support_factor == 1.0:
                picks.append(symbol)

        weight = 1.0 / len(picks) if picks else 0.0
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


def _find_long_term_support(closes: pl.DataFrame) -> dict[str, float]:
    support_levels = {}
    for symbol in closes.columns[1:]:
        min_price = closes[symbol].min()
        support_levels[symbol] = min_price
    return support_levels