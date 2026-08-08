from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines macroeconomic indicators like 3-month rolling GDP growth rate "
        "with sector-specific technical signals such as the 14-day RSI to identify profitable "
        "investment opportunities in the Indian market."
    )

    def __init__(self, rsi_window: int = 14, top_n_rsi: int = 20, gdp_window: int = 3) -> None:
        self._rsi_window = rsi_window
        self._top_n_rsi = top_n_rsi
        self._gdp_window = gdp_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._rsi_window + self._gdp_window)
        if closes.height < self._rsi_window + self._gdp_window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate RSI for each symbol
        rsi_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            history = view.history(lookback=self._rsi_window)
            prices = [float(v) for v in history[symbol].select("close").to_series().drop_nulls().to_list()]
            rsi = _compute_rsi(prices, window=self._rsi_window)
            if rsi:
                rsi_scores[symbol] = rsi[-1]

        # Rank symbols by RSI
        ranked_symbols = sorted(rsi_scores.items(), key=lambda x: x[1], reverse=True)[:self._top_n_rsi]
        
        # Get latest GDP growth rate for overall market health
        gdp_growth_rate = view.history(lookback=self._gdp_window).select("session_date", "adj_close").tail(self._gdp_window)
        if gdp_growth_rate.height < self._gdp_window:
            return Signal(information_available_at=stamp, weights={})

        latest_gdp_growth_rate = float(gdp_growth_rate.select(pl.col("adj_close")[-1]).item())
        if latest_gdp_growth_rate > 0.0:
            valid_symbols = [symbol for symbol, _ in ranked_symbols]
            weight_per_symbol = 1.0 / len(valid_symbols)
            return Signal(
                information_available_at=stamp,
                weights={s: weight_per_symbol for s in valid_symbols}
            )

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_rsi(prices: list[float], window: int) -> list[float]:
    delta = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    up = [v if v > 0 else 0.0 for v in delta]
    down = [-v if v < 0 else 0.0 for v in delta]

    avg_gain = sum(up[-window:]) / window
    avg_loss = sum(down[-window:]) / window

    rs = avg_gain / (avg_loss + 1e-6)  # Avoid division by zero
    rsi = [100 - (100 / (1 + rs))]
    for i in range(window, len(prices)):
        up_i = max(delta[i] - avg_gain, 0)
        down_i = max(-delta[i] - avg_loss, 0)
        avg_gain = (avg_gain * (window - 1) + up_i) / window
        avg_loss = (avg_loss * (window - 1) + down_i) / window
        rs = avg_gain / (avg_loss + 1e-6)
        rsi.append(100 - (100 / (1 + rs)))

    return rsi