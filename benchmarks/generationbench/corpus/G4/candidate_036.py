from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum360d(Strategy):
    rationale = (
        "To exploit cross-sectional momentum in the Indian equity market, this strategy focuses on "
        "identifying stocks that have outperformed their peers over recent periods. By selecting "
        "stocks with the highest returns over a 3-6 month window, we aim to capture continued positive "
        "performance driven by factors such as strong management and favorable industry trends."
    )

    def __init__(self, lookback_days: int = 240) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.height < self._lookback_days or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        symbol_to_returns: dict[str, float] = {}
        for symbol in view.symbols:
            daily_data = history.filter(pl.col("symbol") == symbol)
            if daily_data.height < self._lookback_days / 2:
                continue
            open_prices = [float(v) for v in daily_data["open"].to_list()]
            close_prices = [float(v) for v in daily_data["close"].to_list()]

            # Calculate cumulative returns over the lookback period
            total_return = (close_prices[-1] - open_prices[0]) / open_prices[0]
            symbol_to_returns[symbol] = total_return

        ranked_symbols = sorted(symbol_to_returns, key=symbol_to_returns.get, reverse=True)

        top_20_symbols = ranked_symbols[:20]
        weight_per_symbol = 1.0 / len(top_20_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in top_20_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest