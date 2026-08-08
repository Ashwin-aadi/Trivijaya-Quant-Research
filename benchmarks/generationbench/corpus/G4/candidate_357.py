from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting exploits the empirical observation that stocks with lower "
        "volatility tend to outperform high-volatility stocks over the long term. This is due "
        "to risk-averse investor behavior and market inefficiencies, where low volatility may "
        "persist as a result of reduced risk perception."
    )

    def __init__(self, window: int = 60, portfolio_size: int = 50) -> None:
        self._window = window
        self._portfolio_size = portfolio_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            returns = [(pl.col("close") / pl.col("open").shift(1)).log() - 1.0]
            daily_returns = (
                view.history(lookback=self._window)
                .select(["session_date", symbol])
                .with_columns(returns)
                .sort("session_date")
                .tail(self._window)
                .lazy()
                .select((pl.col(symbol) / pl.col(symbol).shift(1)).log() - 1.0)
                .collect()
            )
            volatilities[symbol] = daily_returns.std()

        ranked_symbols = sorted(volatilities, key=lambda x: volatilities[x])[: self._portfolio_size]
        weights = {s: 1.0 / len(ranked_symbols) for s in ranked_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: weights[s] for s in view.symbols if s in weights},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest