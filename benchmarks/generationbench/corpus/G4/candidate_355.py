from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionCompression(Strategy):
    rationale = (
        "This strategy aims to exploit dispersion or range compression in Indian equity markets. "
        "During dispersion, it identifies undervalued or overvalued stocks for directional bets, "
        "while during range compression, it focuses on mean reversion opportunities."
    )

    def __init__(self, window_dispersion: int = 20, window_range_compression: int = 14, top_n: int = 5) -> None:
        self._window_dispersion = window_dispersion
        self._window_range_compression = window_range_compression
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_range_compression)
        if closes.height < self._window_range_compression:
            return Signal(information_available_at=stamp, weights={})

        dispersion_scores = {}
        range_compression_scores = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window_range_compression + 1:
                continue

            dispersion_scores[symbol] = abs(z_score(prices[-self._window_dispersion:]))
            range_compression_scores[symbol] = atr(prices)

        dispersions = sorted(dispersion_scores.items(), key=lambda x: x[1], reverse=True)
        ranges = sorted(range_compression_scores.items(), key=lambda x: x[1])

        top_dispersion = [symbol for symbol, _ in dispersions[:self._top_n]]
        top_range = [symbol for _, (symbol, _) in ranges[:self._top_n]]

        combined_ranking = combine_rankings(top_dispersion, top_range)

        if not combined_ranking:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(combined_ranking)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in combined_ranking}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def z_score(prices: list[float]) -> float:
    mean_price = sum(prices) / len(prices)
    variance = sum((p - mean_price) ** 2 for p in prices) / (len(prices) - 1)
    std_dev = variance ** 0.5
    return abs((prices[-1] - mean_price) / std_dev)


def atr(prices: list[float]) -> float:
    high_low_diffs = [h - l for h, l in zip(prices[1:], prices[:-1])]
    true_ranges = [
        max(hl, abs(prices[i + 1] - prices[i - 1]), abs(prices[i + 1] - prices[i])) 
        for i, hl in enumerate(high_low_diffs)
    ]
    return sum(true_ranges) / len(true_ranges)


def combine_rankings(dispersion: list[str], range_compression: list[str]) -> list[str]:
    combined = sorted(set(dispersion + range_compression), key=lambda x: (dispersion.index(x) if x in dispersion else float("inf"), range_compression.index(x) if x in range_compression else float("inf")))
    return combined[:min(len(dispersion) + len(range_compression))]