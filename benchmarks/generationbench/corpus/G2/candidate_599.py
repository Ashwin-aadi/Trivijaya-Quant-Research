from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumAndVolatility(Strategy):
    rationale = (
        "Stocks with high momentum tend to continue their recent trends. However, high volatility "
        "can indicate a lack of direction or increased risk. By combining these two characteristics, "
        "we can identify stocks that are trending strongly but have not yet experienced excessive "
        "volatility."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 10) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window)
        if closes.height < self._momentum_window:
            return Signal(information_available_at=stamp, weights={})

        history = view.history(lookback=max(self._momentum_window, self._volatility_window))
        volatility = _calculate_volatility(history)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in volatility.columns:
                continue
            momentum = float(closes[symbol].drop_nulls().to_list()[-1]) - float(
                closes[symbol].drop_nulls().to_list()[0]
            )
            vol = float(volatility[symbol].drop_nulls().to_list()[-1])
            if vol < 2.0 and momentum > 0:
                picks.append(symbol)

        picks = picks[:5]  # Select top 5 symbols
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(history: pl.DataFrame) -> pl.DataFrame:
    symbols = history.columns[2:]  # Skip the first two columns (symbol and session_date)
    volatility = (
        history.select(symbols)
        .with_columns(
            [(pl.col(symbol).std().alias(symbol)) for symbol in symbols]
        )
        .sort("session_date", descending=True)
        .head(self._volatility_window)
    )
    return volatility