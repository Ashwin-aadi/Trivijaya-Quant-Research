from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term and long-term momentum can provide a more robust signal. "
        "Short-term momentum suggests recent strength, while long-term momentum indicates sustained"
        "performance."
    )

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._long_window)
        if closes.height < self._long_window:
            return Signal(information_available_at=stamp, weights={})

        short_mom = (closes[f"close_{self._short_window}"].to_list()[-1] / 
                     closes[f"close_{self._short_window}"].shift(1).to_list()[0] - 1.0)
        long_mom = (closes[f"close_{self._long_window}"].to_list()[-1] /
                    closes[f"close_{self._long_window}"].shift(1).to_list()[0] - 1.0)

        short_ranks = pl.Series(short_mom).rank(method="ordinal", descending=True).to_list()
        long_ranks = pl.Series(long_mom).rank(method="ordinal", descending=True).to_list()

        picks: list[str] = []
        for symbol in view.symbols:
            if (short_ranks[-1][symbol] + long_ranks[-1][symbol]) >= len(view.symbols) / 2:
                picks.append(symbol)

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