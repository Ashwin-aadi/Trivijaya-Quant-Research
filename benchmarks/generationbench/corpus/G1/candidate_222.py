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

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        relative_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate daily returns
            returns = [(values[i] / values[i - 1] - 1.0) for i in range(1, self._window)]
            mean_return = sum(returns) / len(returns)
            relative_strengths[symbol] = mean_return

        top_n_symbols = sorted(relative_strengths.items(), key=lambda x: x[1], reverse=True)[:5]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        total_weight = 0.0
        signal_weights = {symbol: (weight / sum(v for _, v in top_n_symbols)) for symbol, weight in top_n_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in signal_weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest