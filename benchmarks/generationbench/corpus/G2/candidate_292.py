from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price reversion occurs when a stock that has deviated significantly from its mean "
        "price level tends to return towards it over time. By identifying such deviations, we "
        "can generate trading signals based on the price levels of stocks relative to their "
        "moving averages."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        average_price = (
            closes[[col for col in closes.columns if col != "session_date"]]
            .mean()
            .to_list()
        )
        deviation_scores = {
            symbol: (close - avg) / avg
            for symbol, close, avg in zip(
                closes["session_date"].to_list(),
                closes[[col for col in closes.columns if col != "session_date"]].to_list().transpose()[0],
                average_price,
            )
        }

        sorted_symbols = [
            sym
            for _, (sym, _) in sorted(deviation_scores.items(), key=lambda item: abs(item[1]), reverse=True)
            if abs(deviation_scores[sym]) > 2.0 / len(closes.columns)  # Threshold to consider significant deviation
        ]

        top_n_symbols = sorted_symbols[:5]
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