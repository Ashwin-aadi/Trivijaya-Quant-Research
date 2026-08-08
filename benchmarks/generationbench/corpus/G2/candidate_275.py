from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "We hypothesize that stocks with low relative volatility and high recent positive momentum "
        "are likely to continue trending. Low relative volatility suggests that the stock is less "
        "susceptible to large price swings, while strong recent momentum implies a higher probability of continued outperformance."
    )

    def __init__(self, window_volatility: int = 20, window_momentum: int = 5) -> None:
        self._window_volatility = window_volatility
        self._window_momentum = window_momentum

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_momentum + self._window_volatility)
        if closes.height < self._window_volatility + self._window_momentum:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the volatility using 20-day rolling standard deviation
        volatilities = []
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_volatility + self._window_momentum - 1:
                continue

            # Compute the rolling standard deviation
            vol = (
                pl.DataFrame({"close": values})
                .rolling_window(window=self._window_volatility, closed="both")
                .std()
                .to_series()
                .drop_nulls()
                .to_list()[-1]
            )
            if not isinstance(vol, (float, int)):
                continue
            volatilities.append((symbol, vol))

        # Filter symbols based on low relative volatility
        filtered_symbols = [symbol for symbol, _ in sorted(volatilities) if _ <= 0.1]

        # Calculate the momentum using 5-day closing price change
        momentums = []
        for symbol in filtered_symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_momentum + 1:
                continue

            # Compute the recent momentum
            mom = (values[-1] - values[-self._window_momentum]) / values[-self._window_momentum]
            momentums.append((symbol, mom))

        # Filter symbols based on high positive momentum
        filtered_symbols_and_moments = [s for s, m in sorted(momentums) if m > 0.1]
        if not filtered_symbols_and_moments:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols_and_moments)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols_and_moments},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest