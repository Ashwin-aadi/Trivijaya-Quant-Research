from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Investing in securities with the strongest relative performance can lead to higher "
        "returns compared to a broader market index. This strategy selects symbols that have "
        "outperformed their peers over a recent period."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        relative_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) >= self._window:

                # Calculate daily returns
                returns = [(values[i] / values[i - 1] - 1.0) for i in range(1, self._window)]
                mean_return = sum(returns) / len(returns)

                relative_strengths[symbol] = mean_return

        top_n_symbols = sorted(relative_strengths.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in top_n_symbols]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest