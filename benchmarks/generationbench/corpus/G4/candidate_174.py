from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RSIRelativeStrength(Strategy):
    rationale = (
        "This strategy exploits the relative strength of stocks compared to the broader market index "
        "by selecting those with a higher Relative Strength Index (RSI). It leverages the persistence of "
        "strong stock performance in trending markets while maintaining diversification and risk management."
    )

    def __init__(self, window: int = 14, threshold: float = 70, n_stocks: int = 30) -> None:
        self._window = window
        self._threshold = threshold
        self._n_stocks = n_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty50_history = history.filter(pl.col("symbol").is_in(view.symbols))
        individual_stocks = [s for s in view.symbols if s not in ["NIFTY 50", "NIFTY NEXT 50"]]
        nifty50_close = _compute_nifty50_close(nifty50_history)

        rsi_individual = {}
        for symbol in individual_stocks:
            rsi, close_prices = _calculate_rsi(history.filter(pl.col("symbol") == symbol), self._window)
            if rsi[-1] > nifty50_close[-1] * (self._threshold / 100):
                rsi_individual[symbol] = close_prices

        selected_stocks = sorted(rsi_individual.keys(), key=lambda s: rsi_individual[s][-1], reverse=True)[: self._n_stocks]
        if not selected_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_nifty50_close(history: pl.DataFrame) -> list[float]:
    nifty50_history = history.filter(pl.col("symbol") == "NIFTY 50")
    closes = nifty50_history.select("close").to_numpy().flatten().tolist()
    assert all(isinstance(c, float) for c in closes)
    return closes


def _calculate_rsi(history: pl.DataFrame, window: int) -> tuple[list[float], list[float]]:
    close_prices = history.select("adj_close").to_numpy().flatten().tolist()

    delta = [close - open for open, close in zip(close_prices[:-1], close_prices[1:])]
    gain = [d if d > 0 else 0 for d in delta]
    loss = [-d if d < 0 else 0 for d in delta]

    avg_gain = sum(gain[:window]) / window
    avg_loss = abs(sum(loss[:window])) / window

    rs = avg_gain / avg_loss
    rsi = [100 - (100 / (1 + rs))]

    for i in range(1, len(delta)):
        avg_gain = ((avg_gain * (window - 1)) + gain[i]) / window
        avg_loss = ((avg_loss * (window - 1)) + loss[i]) / window

        if avg_loss == 0:
            rs = float("inf")
        else:
            rs = avg_gain / avg_loss

        rsi.append(100 - (100 / (1 + rs)))

    return rsi, close_prices