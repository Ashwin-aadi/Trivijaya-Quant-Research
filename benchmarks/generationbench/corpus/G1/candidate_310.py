from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit stronger performance during specific "
        "seasons of the year. This strategy aims to identify such seasonal effects and "
        "capitalize on them by allocating capital towards symbols that perform well in their "
        "historically strongest season."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 365)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        seasonal_signals: dict[str, float] = {}
        
        for symbol in symbols:
            if symbol not in history.symbol.unique().to_list():
                continue

            df = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            adj_closes = df["adj_close"].to_list()

            seasonal_returns: dict[int, float] = {}
            for i in range(12):
                year_data = [adj_closes[j] for j in range(len(adj_closes)) if (j - 1) % 12 == i]
                if len(year_data) < self._window:
                    continue
                max_return = max((adj_closes[i + self._window] / adj_closes[i]) - 1.0 for i in range(0, len(adj_closes) - self._window))
                seasonal_returns[i] = max_return

            strongest_season = max(seasonal_returns, key=seasonal_returns.get)
            if seasonal_returns[strongest_season] > 0:
                seasonal_signals[symbol] = seasonal_returns[strongest_season]

        if not seasonal_signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / sum(seasonal_signals.values())
        weighted_signals = {symbol: seasonal_signals[symbol] * weight for symbol in seasonal_signals}

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weighted_signals.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest