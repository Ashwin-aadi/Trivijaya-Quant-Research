from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy identifies stocks with strong relative performance compared to the broader market universe. "
        "By focusing on the Relative Strength Index (RSI) over a 6-month lookback period, it selects top N stocks for long positions, "
        "leveraging the persistence in stock selection outperformance."
    )

    def __init__(self, window: int = 180, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        benchmark_close = view.latest_close()["^NIFTY 100"]
        rsi_values: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol == "^NIFTY 100":
                continue
            adj_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            benchmark_adj_closes = [
                float(benchmark_close[i]) for i in range(len(adj_closes))
            ]
            rsi_values[symbol] = _compute_rsi(
                adjusted_closes=adj_closes, benchmark_closes=benchmark_adj_closes
            )

        top_n_symbols = sorted(rsi_values.items(), key=lambda item: -item[1])[: self._top_n]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s, _ in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_rsi(adjusted_closes: list[float], benchmark_closes: list[float]) -> float:
    gains = [a - b for a, b in zip(adjusted_closes[1:], benchmark_closes[:-1])]
    losses = [-g for g in gains]
    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)

    if avg_loss == 0:
        return float("inf")

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi