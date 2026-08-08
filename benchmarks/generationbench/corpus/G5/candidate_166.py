from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityEffect(Strategy):
    rationale = (
        "Seasonal effects can be exploited by identifying stocks that exhibit higher returns "
        "at certain times of the year. This strategy aims to capture these anomalies by "
        "allocating capital to stocks with historically strong performance during their peak "
        "seasons, based on log returns."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        symbols = view.symbols
        seasonal_effect = pl.DataFrame()

        for symbol in symbols:
            if symbol not in closes.columns:
                continue

            symbol_history = history.select(
                [pl.col("session_date"), pl.col(symbol)]
            ).filter(pl.col("session_date").dt.month().is_in([6, 7, 8]))

            if symbol_history.height < self._window:
                continue

            log_returns = (
                symbol_history[symbol]
                .drop_nulls()
                .to_list()[1:]
                + [float(symbol_history[symbol].mean())] * (self._window - len(closes))
            )
            mean_log_return = sum(log_returns) / self._window
            if pl.col("symbol").is_nan().any():
                continue

            seasonal_effect = seasonal_effect.with_columns(
                pl.Series(symbol, [float(mean_log_return)] * self._window)
            )

        if seasonal_effect.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scores = []
        for symbol in symbols:
            symbol_closes = closes[symbol].drop_nulls().to_list()
            log_returns = [
                float(symbol_closes[i] / symbol_closes[i - 1] - 1.0)
                for i in range(1, len(symbol_closes))
            ]
            score = sum(
                1
                for i in range(self._window)
                if log_returns[i] >= seasonal_effect[symbol][i]
            )
            scores.append((symbol, score))

        top_symbols = [s[0] for s in sorted(scores, key=lambda x: -x[1])[:5]]
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