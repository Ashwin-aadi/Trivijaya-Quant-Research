from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks based on their relative strength using the RSI "
        "indicator. Stocks with a lower RSI indicating oversold conditions and outperformance "
        "against NIFTY 50 are longed to capitalize on potential reversals."
    )

    def __init__(self, window_rsi: int = 14, threshold_oversold: float = 30, lookback_performance: int = 30) -> None:
        self._window_rsi = window_rsi
        self._threshold_oversold = threshold_oversold
        self._lookback_performance = lookback_performance

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_performance + 1).sort("session_date", descending=True)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty50_closes = history.filter(pl.col("symbol").is_in(view.symbols))
        symbols = [s for s in view.symbols if s not in ["NIFTY 50"]]
        
        rsi_data = _calculate_rsi(nifty50_closes, self._window_rsi)
        performance_data = _calculate_performance(rsi_data, history)

        eligible_symbols = []
        for symbol in symbols:
            if (performance_data[symbol] >= 1) and (rsi_data[symbol][-1] < self._threshold_oversold):
                eligible_symbols.append(symbol)

        eligible_symbols = eligible_symbols[:20]
        if not eligible_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(eligible_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in eligible_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_rsi(df: pl.DataFrame, window: int) -> dict[str, float]:
    rsi_values = {}
    for symbol in df["symbol"].unique().to_list():
        values = [float(v) for v in df.filter(pl.col("symbol") == symbol)["adj_close"].drop_nulls().to_list()]
        if len(values) < window:
            continue
        delta = [v2 - v1 for v1, v2 in zip(values[:-1], values[1:])]
        gain = [max(v, 0.0) for v in delta]
        loss = [-min(v, 0.0) for v in delta]
        avg_gain = sum(gain[-window:]) / window
        avg_loss = sum(loss[-window:]) / window
        rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
        rsi = 100 - (100 / (1 + rs))
        rsi_values[symbol] = rsi
    return rsi_values


def _calculate_performance(rsi_data: dict[str, float], history: pl.DataFrame) -> dict[str, float]:
    performance_data = {}
    for symbol in rsi_data.keys():
        nifty50_closes = history.filter(pl.col("symbol") == "NIFTY 50")
        if nifty50_closes.is_empty() or symbol not in rsi_data:
            continue
        nifty50_values = [float(v) for v in nifty50_closes["adj_close"].drop_nulls().to_list()]
        stock_values = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].drop_nulls().to_list()]
        if len(nifty50_values) < 30 or len(stock_values) < 30:
            continue
        nifty50_returns = sum([(n2 - n1) / n1 for n1, n2 in zip(nifty50_values[:-30], nifty50_values[1:-29])])
        stock_returns = sum([(s2 - s1) / s1 for s1, s2 in zip(stock_values[:-30], stock_values[1:-29])])
        performance_data[symbol] = stock_returns / nifty50_returns if nifty50_returns != 0 else 0.0
    return performance_data