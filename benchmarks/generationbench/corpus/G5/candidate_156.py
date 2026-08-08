from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy selects the top-performing stocks over a lookback period based on "
        "return. Stocks with higher returns are expected to continue outperforming due to "
        "momentum effects."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        daily_returns = (
            history
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .select(pl.exclude(["symbol", "session_date"])).to_numpy()
        )

        if not daily_returns.size:
            return Signal(information_available_at=stamp, weights={})

        symbol_returns = {symbol: [] for symbol in view.symbols}
        for i, row in enumerate(daily_returns):
            for j, symbol in enumerate(view.symbols):
                symbol_returns[symbol].append(row[j])

        top_performers = []
        for symbol, returns in symbol_returns.items():
            if len(returns) < self._window:
                continue
            mean_return = sum(returns[-self._window:]) / len(returns)
            if mean_return >= max(mean_return for mean_return in symbol_returns.values()):
                top_performers.append(symbol)

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_performers}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest