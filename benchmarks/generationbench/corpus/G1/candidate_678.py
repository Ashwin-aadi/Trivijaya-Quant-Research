from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks that have outperformed the market in recent history. "
        "The idea is to buy stocks with strong relative performance, assuming they are undervalued and likely to continue outperforming."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in closes.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        returns = {s: [] for s in symbols}
        benchmark_returns = []

        for symbol in symbols:
            daily_closes = [
                float(v) for v in closes[symbol].drop_nulls().to_list()
            ]
            if len(daily_closes) < self._window + 1:
                continue
            returns[symbol] = [daily_closes[i + 1] / daily_closes[i] - 1.0
                               for i in range(len(daily_closes) - 1)]
            benchmark_returns.extend([float(v) for v in view.closes(lookback=self._window)[symbol].drop_nulls().to_list()[-self._window:] if v != 0])

        avg_returns = {s: sum(r) / len(r) for s, r in returns.items()}
        bench_avg_return = sum(benchmark_returns) / len(benchmark_returns)
        relative_strengths = {
            s: (avg_returns[s] - bench_avg_return) / bench_avg_return
            if bench_avg_return != 0 else float("inf")
            for s in symbols
        }

        top_n_symbols = sorted(relative_strengths, key=lambda k: relative_strengths[k], reverse=True)[:self._top_n]

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest