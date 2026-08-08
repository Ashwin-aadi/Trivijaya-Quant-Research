from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed the broader market over a certain period "
        "are likely to continue outperforming due to momentum effects."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        if any(symbol not in closes.columns for symbol in view.symbols):
            return Signal(information_available_at=stamp, weights={})

        def compute_rsi(df: pl.DataFrame) -> float:
            deltas = df["adj_close"].to_list()[1:] - df["adj_close"].to_list()[:-1]
            gains = [g if g > 0 else 0 for g in deltas]
            losses = [-l if l < 0 else 0 for l in deltas]

            avg_gain = sum(gains) / len(gains)
            avg_loss = sum(losses) / len(losses)

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        symbols_rsi: dict[str, float] = {}
        for symbol in view.symbols:
            df = history.select(["session_date", pl.col(symbol).alias("adj_close")])
            if not df.height or any(pl.isnan(val) for val in df["adj_close"].to_list()):
                continue
            rsi = compute_rsi(df)
            symbols_rsi[symbol] = rsi

        sorted_symbols = sorted(symbols_rsi.items(), key=lambda x: x[1], reverse=True)
        top_n_symbols = [s for s, _ in sorted_symbols[:5]]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

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