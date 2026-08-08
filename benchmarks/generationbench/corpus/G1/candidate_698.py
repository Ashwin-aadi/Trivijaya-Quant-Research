from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends by scaling the trend signal with historical volatility. "
        "High volatility periods dampen the trend signal to avoid overtrading, while low volatility "
        "periods amplify the trend signal for more aggressive trading."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol)
            if df.is_empty():
                continue
            open_prices = [float(v) for v in df["open"].to_list()]
            close_prices = [float(v) for v in df["close"].to_list()]
            returns = [(c - o) / o for o, c in zip(open_prices[:-1], close_prices[1:])]

            mean_return = sum(returns) / len(returns)
            volatility = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5

            if volatility == 0:
                continue
            recent_close = float(df.filter(pl.col("session_date") == stamp).select("adj_close").to_list()[0][0])
            trend_signal = (recent_close - open_prices[-1]) / (close_prices[1] - open_prices[0])

            scaled_signal = abs(trend_signal) * volatility
            symbol_data[symbol] = scaled_signal

        top_symbols = sorted(symbol_data, key=symbol_data.get, reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest