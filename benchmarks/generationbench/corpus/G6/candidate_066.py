from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength14d(Strategy):
    rationale = (
        "This strategy identifies relative strength by using the RSI indicator over a 14-day "
        "period. Stocks are bought when the RSI falls below 30 for two consecutive days and "
        "sold when it rises above 70 for three consecutive days. The aim is to capitalize on "
        "momentum shifts in the market."
    )

    def __init__(self, window: int = 14, overbought_threshold: float = 70, oversold_threshold: float = 30, max_positions: int = 50) -> None:
        self._window = window
        self._overbought_threshold = overbought_threshold
        self._oversold_threshold = oversold_threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < 2 * (self._window + 1):
            return Signal(information_available_at=stamp, weights={})

        signal = {}
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date").select(
                pl.col("adj_close"),
                (pl.col("adj_close").shift(-self._window) / pl.col("adj_close") - 1.0).alias("returns")
            )
            rsi = _compute_rsi(df, self._window)
            if len(rsi) < 2:
                continue
            if all(r <= self._oversold_threshold for r in rsi[-2:]) and signal.get(symbol, False):
                signal[symbol] = "BUY"
            elif any(r >= self._overbought_threshold for r in rsi[-3:]):
                if symbol in signal and signal[symbol] == "BUY":
                    signal.pop(symbol)
                else:
                    continue
            elif not signal.get(symbol) and all(r <= self._oversold_threshold for r in rsi[-2:]) and any(r >= self._oversold_threshold - 5 for r in rsi[-3:]):
                signal[symbol] = "BUY"
        positions = sorted(signal.keys(), key=lambda x: (signal[x], view.latest_close()[x]), reverse=True)[:self._max_positions]
        if not positions:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(positions)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in positions}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_rsi(df: pl.DataFrame, window: int) -> list[float]:
    delta = df.select(pl.col("returns")).to_numpy().flatten()
    gains = [0.0] + [v for v in delta[1:] if v > 0]
    losses = [0.0] + [-v for v in delta[1:] if v < 0]
    avg_gain = sum(gains[-window:]) / window
    avg_loss = sum(losses[-window:]) / window
    rs = avg_gain / abs(avg_loss) if avg_loss != 0 else float("inf")
    rsi = [50 - (100 / (1 + rs))]
    for i in range(1, len(delta)):
        gain = delta[i] if delta[i] > 0 else 0
        loss = -delta[i] if delta[i] < 0 else 0
        avg_gain = ((avg_gain * (window - 1)) + gain) / window
        avg_loss = ((avg_loss * (window - 1)) + loss) / window
        rs = avg_gain / abs(avg_loss) if avg_loss != 0 else float("inf")
        rsi.append(50 - (100 / (1 + rs)))
    return rsi