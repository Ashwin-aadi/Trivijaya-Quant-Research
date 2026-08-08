from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency of stocks within a market to continue "
        "their recent price trends. By identifying and investing in the top-performing stocks, "
        "we aim to capture their momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_close = view.latest_close()
        top_symbols = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            adj_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue
            returns = [(adj_closes[i] / adj_closes[i-1] - 1.0) for i in range(1, len(adj_closes))]
            mean_return = sum(returns) / len(returns)
            if mean_return >= max(mean_return for s in view.symbols if s != symbol):
                top_symbols.append(symbol)

        top_symbols = top_symbols[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest