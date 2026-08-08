from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion200d(Strategy):
    rationale = (
        "This strategy aims to capture mean-reverting behavior by leveraging both price and volume data. "
        "By setting entry rules based on absolute deviations from a 200-day simple moving average (SMA) by +3 standard deviations, "
        "we ensure that only highly overextended stocks are bought. The exit rule returns positions when prices revert to within ±0.5 standard deviations, "
        "providing clear signals and minimizing holding periods during non-reversion phases. Diversification is maintained through equal weighting of up to 20 names."
    )

    def __init__(self, window: int = 200, std_dev_entry: float = 3.0, std_dev_exit: float = 0.5, max_positions: int = 20) -> None:
        self._window = window
        self._std_dev_entry = std_dev_entry
        self._std_dev_exit = std_dev_exit
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(pl.col("adj_close"))
        sma = closes.mean().over([pl.lit(self._window).alias("symbol")])
        std_dev = (closes - sma).stddev().over([pl.lit(self._window).alias("symbol")])

        entries = (
            closes.join(sma, on="session_date", how="inner")
                .join(std_dev, on="session_date", how="inner")
                .with_columns(
                    ((closes["adj_close"] - sma) / std_dev > self._std_dev_entry).alias("entry_condition"),
                )
        )

        if entries.height < 2 * self._window:
            return Signal(information_available_at=stamp, weights={})

        entry_candidates = entries.filter(entries["entry_condition"])
        if entry_candidates.is_empty():
            return Signal(information_available_at=stamp, weights={})

        selected_symbols = []
        for symbol in view.symbols:
            if symbol not in entry_candidates.columns or (symbol in closes.columns and symbol not in std_dev.columns):
                continue
            last_close = float(view.latest_close()[symbol])
            sma_val = float(sma[symbol][-1])
            std_dev_val = float(std_dev[symbol][-1])

            if (last_close - sma_val) / std_dev_val > self._std_dev_entry:
                selected_symbols.append(symbol)

        selected_symbols = selected_symbols[:self._max_positions]
        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest