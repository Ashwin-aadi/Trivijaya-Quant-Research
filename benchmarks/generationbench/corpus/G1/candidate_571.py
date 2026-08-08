from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression signals periods where the price action has been unusually tight. "
        "This can indicate a potential reversal or continuation of trend based on historical volatility."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_values = [float(v) for v in history[symbol + "_high"].to_list()]
            low_values = [float(v) for v in history[symbol + "_low"].to_list()]

            # Calculate daily range
            ranges = [h - l for h, l in zip(high_values, low_values)]
            mean_range = sum(ranges) / len(ranges)
            std_dev_range = pl.Series(ranges).std()
            if std_dev_range > 0:
                normalized_ranges = [(r - mean_range) / std_dev_range for r in ranges]
                range_compression_score = abs(min(normalized_ranges))
                range_compression_scores[symbol] = range_compression_score

        if not range_compression_scores:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = [
            symbol
            for _, symbol in sorted(
                range_compression_scores.items(), key=lambda item: -item[1]
            )
        ][:5]

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest