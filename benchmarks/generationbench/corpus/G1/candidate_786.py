from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength to the NIFTY 100 index are expected to outperform "
        "the broader market. This strategy identifies such stocks based on their recent price "
        "performance compared to the overall market."
    )

    def __init__(self, window: int = 20, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window).select(["symbol", "session_date", pl.col("adj_close").alias("close")])

        # Calculate the average close for NIFTY 100 symbols
        nifty_100_history = history.filter(pl.col("symbol").is_in(view.symbols))
        nifty_100_closes = nifty_100_history.select("adj_close")
        avg_nifty_100_close = (nifty_100_closes.sum() / nifty_100_history.height).item()

        # Calculate relative strength for each symbol
        relative_strength: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_values) < self._window:
                continue

            last_close = close_values[-1]
            relative_strength[symbol] = last_close / avg_nifty_100_close

        # Select symbols with a relative strength above the threshold
        strong_symbols = [symbol for symbol, rs in relative_strength.items() if rs >= self._threshold]

        if not strong_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(strong_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in strong_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest