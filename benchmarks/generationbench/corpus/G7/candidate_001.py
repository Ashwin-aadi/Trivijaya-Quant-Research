from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Short-horizon mean reversion looks for stocks that have deviated significantly from "
        "their recent average price levels. If a stock's price is below its 5-day moving "
        "average, it suggests the stock may revert to its mean in the near term."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            closes.select(pl.col("adj_close").mean().alias(f"avg_{self._window}d"))
            .to_dict(True)[0][f"avg_{self._window}d"]
        )

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            latest_close = float(values[-1])
            mean_reversion_signal = (latest_close - avg_close) / avg_close
            signals[symbol] = mean_reversion_signal

        sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        top_n_symbols = [symbol for symbol, _ in sorted_signals[:5]]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest