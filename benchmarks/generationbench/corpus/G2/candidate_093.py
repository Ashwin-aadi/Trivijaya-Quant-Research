from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks are less prone to sudden price movements and may offer more stable returns. "
        "By tilting the portfolio towards low-volatility stocks, we can reduce overall risk and potentially improve return consistency."
    )

    def __init__(self, lookback_window: int = 60) -> None:
        self._lookback_window = lookback_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(symbol) for symbol in view.symbols]
        volatilities = _calculate_volatility(symbols, history)

        sorted_symbols = [
            s for _, s in sorted(volatilities.items(), key=lambda item: item[1])
        ][:5]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(symbols: list[str], history: pl.DataFrame) -> dict[str, float]:
    volatilities = {}
    for symbol in symbols:
        close_prices = [float(v) for v in history.select(pl.col(symbol))["close"].to_list()]
        if len(close_prices) < 2:
            continue
        log_returns = [
            (np.log(prices[i + 1] / prices[i]) * 100)
            for i, _ in enumerate(prices[:-1])
        ]
        volatility = np.std(log_returns) * np.sqrt(252)
        volatilities[symbol] = volatility
    return volatilities