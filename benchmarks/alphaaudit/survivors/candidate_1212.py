from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum identifies stocks with the strongest recent price "
        "performance. Investing in these top performers can lead to higher returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width == 0:
            return Signal(information_available_at=stamp, weights={})

        top_n = min(len(view.symbols), 5)
        ranks: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            rank = (
                pl.Series(values)
                .rank(method="dense", descending=True)
                .to_list()[0]
            )
            ranks[symbol] = rank

        sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item: item[1])}
        top_symbols = list(sorted_ranks.keys())[:top_n]

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