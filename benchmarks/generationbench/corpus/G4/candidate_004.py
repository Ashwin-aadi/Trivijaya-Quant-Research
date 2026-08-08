from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "This strategy aims to capitalize on mean-reverting behavior in stock prices within the Indian market. "
        "By identifying stocks that have fallen below their trailing reference levels and showing signs of improvement, "
        "it seeks to benefit from the tendency for prices to revert towards historical averages."
    )

    def __init__(self, window_50: int = 50, window_200: int = 200, top_n: int = 10) -> None:
        self._window_50 = window_50
        self._window_200 = window_200
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_200)
        if closes.height < self._window_200:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            sma_50 = sum(values[-self._window_50:]) / self._window_50
            sma_200 = sum(values[-self._window_200:]) / self._window_200

            if values[-1] < min(sma_50, sma_200) and \
                    (sma_50 > sma_200 or
                     _rsi_check(view.history(lookback=self._window_200), symbol) >= 30):
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


def _rsi_check(view_history: pl.DataFrame, symbol: str) -> float:
    history = view_history.select(pl.col(symbol))
    if history.height < 20:
        return 50.0
    closes = [float(v) for v in history.to_list()[0]]
    delta = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gain = sum([d if d > 0 else 0 for d in delta])
    loss = sum([-d if d < 0 else 0 for d in delta])

    avg_gain = gain / (len(delta) - 1)
    avg_loss = abs(loss) / (len(delta) - 1)

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1 + rs))
    return rsi