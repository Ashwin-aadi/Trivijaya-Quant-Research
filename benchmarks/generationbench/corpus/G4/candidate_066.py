from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks that outperform the NIFTY 100 index over a 6-month period "
        "based on their relative strength. Strong companies tend to persist in outperforming due to "
        "their superior fundamentals and competitive advantages."
    )

    def __init__(self, window: int = 180, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        benchmark_close = view.closes(lookback=None)["^NIFTY 100"].to_list()
        symbols = [symbol for symbol in view.symbols if symbol != "^NIFTY 100"]

        rsi_values: dict[str, float] = {}
        for symbol in symbols:
            data = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            prices = data.select(
                pl.col("adj_close").to_list(), pl.col("close").to_list()
            )
            close_prices = [float(v) for v in prices[0].to_list()]
            benchmark_closes = [
                float(v)
                for v in history.filter(pl.col("symbol") == "^NIFTY 100")
                                  .select("adj_close")
                                  .sort(by="session_date")
                                  .select(["adj_close"])
                                  .drop_nulls()
                                  ["adj_close"]
                                  .to_list()
            ]
            rsi = _compute_rsi(close_prices, benchmark_closes)
            rsi_values[symbol] = rsi

        sorted_symbols = [k for k, v in sorted(rsi_values.items(), key=lambda item: -item[1])]
        top_decile = sorted_symbols[: self._top_n]

        if not top_decile:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_decile)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_decile},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_rsi(prices: list[float], benchmark_prices: list[float]) -> float:
    deltas = [p - benchmark_prices[i - 1] for i, p in enumerate(benchmark_prices[1:], start=1)]
    gains = [g if g > 0 else 0 for g in deltas]
    losses = [-l if l > 0 else 0 for l in deltas]

    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)

    rs = avg_gain / (avg_loss + 1e-6)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi