from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reversion occurs when a security that has been overvalued (or undervalued) "
        "tends to move towards its mean value. By tracking the trailing average of prices and "
        "identifying deviations from this average, we can find securities that are likely to "
        "revert to their mean price levels."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_price = closes.mean().to_dict()[0]["adj_close"]
        deviations = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            price_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            latest_price = price_series[-1]
            deviation = (latest_price - avg_price) / avg_price
            deviations[symbol] = deviation

        sorted_deviations = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)
        top_n_symbols = [s for s, d in sorted_deviations[:5]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest