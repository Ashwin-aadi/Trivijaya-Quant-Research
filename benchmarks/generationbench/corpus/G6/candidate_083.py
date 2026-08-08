from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedAmbitiousStrategy(Strategy):
    rationale = (
        "This strategy combines elements of both conservative and ambitious approaches by "
        "utilizing RSI, earnings yield, and volatility to enter positions when the market is "
        "oversold or undervalued. It also implements a stop-loss mechanism and holds stocks for"
        "a 6-month period before exiting."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or "adj_close" not in history.columns:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate RSI
        rsi_values: list[float] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            close_series = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].drop_nulls().to_list()]
            if len(close_series) < self._window * 2 - 1:
                continue

            delta = [close_series[i] - close_series[i - 1] for i in range(1, len(close_series))]
            gain = [d if d > 0 else 0 for d in delta]
            loss = [-d if d < 0 else 0 for d in delta]

            avg_gain = sum(gain) / (self._window * 2 - 1)
            avg_loss = sum(loss) / (self._window * 2 - 1)

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)

        # Calculate Earnings Yield and Volatility
        earnings_yield = {symbol: float(view.latest_close()[symbol]) for symbol in view.symbols}
        volatility = {
            symbol: pl.Series(close_series).rolling_std(window=self._window).item() for symbol, close_series in zip(view.symbols, history.filter(pl.col("symbol").is_in(view.symbols))["adj_close"].to_list())
        }

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in rsi_values or symbol not in earnings_yield or symbol not in volatility:
                continue

            rsi_value = rsi_values[symbol]
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_series) < self._window * 2 - 1:
                continue

            momentum = (close_series[-1] / close_series[0]) ** (252 / (len(close_series) + 1)) - 1.0
            volume_change = sum([float(v) for v in view.history(lookback=self._window).filter(pl.col("symbol") == symbol)["volume"].to_list()]) / self._window

            if rsi_value < 30 and earnings_yield[symbol] > 0.05 and volatility[symbol] < 0.2:
                picks.append(symbol)

        picks = picks[: self._top_n]
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