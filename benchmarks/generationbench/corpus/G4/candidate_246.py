from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy identifies and invests in stocks with strong relative performance against "
        "the Nifty 50 index. It leverages the persistence of outperformance driven by behavioral "
        "finance theories to capture residual returns while maintaining a balanced risk profile."
    )

    def __init__(self, window: int = 90, top_percentile: float = 0.3) -> None:
        self._window = window
        self._top_percentile = top_percentile

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        nifty_50_closes = view.closes(lookback=self._window).select(
            [pl.col("NIFTY50").alias("close")]
        )
        if nifty_50_closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        relative_strength: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol != "NIFTY50":
                stock_close = closes[symbol].to_list()
                nifty_50_close = nifty_50_closes["close"].to_list()
                if len(stock_close) < self._window or len(nifty_50_close) < self._window:
                    continue

                stock_returns = [(stock_close[i] / stock_close[i - 1]) - 1.0 for i in range(1, len(stock_close))]
                nifty_50_returns = [(nifty_50_close[i] / nifty_50_close[i - 1]) - 1.0 for i in range(1, len(nifty_50_close))]

                stock_cumulative_return = (1 + sum(stock_returns)).item()
                nifty_50_cumulative_return = (1 + sum(nifty_50_returns)).item()

                relative_strength[symbol] = stock_cumulative_return / nifty_50_cumulative_return

        ranked_symbols = sorted(relative_strength.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in ranked_symbols[:int(len(view.symbols) * self._top_percentile)]]

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