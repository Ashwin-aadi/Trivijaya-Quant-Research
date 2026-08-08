from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy identifies stocks with higher relative strength index (RSI) compared to the broader market "
        "universe. By focusing on outperforming stocks, it aims to capitalize on short-term momentum driven by positive"
        " investor sentiment and herd behavior."
    )

    def __init__(self, window: int = 14, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        nifty50_close = history.select(pl.col("adj_close").filter(pl.col("symbol") == "NIFTY50")).to_series().to_list()[0]
        nifty50_returns = [float(nifty50_close[i]) / float(nifty50_close[i - 1]) - 1.0 for i in range(1, len(nifty50_close))]

        symbols = [symbol for symbol in view.symbols if symbol != "NIFTY50"]
        closes = history.select(pl.col("adj_close").filter(pl.col("symbol").is_in(symbols))).transpose().to_series().to_list()

        rsi_scores: list[float] = []
        for i, close_prices in enumerate(closes):
            if len(close_prices) < self._window:
                continue

            close_series = pl.Series(close_prices)
            delta = close_series.diff()
            gain = (delta.where(delta > 0)).mean().to_numpy()[0]
            loss = (-delta.where(delta < 0)).mean().to_numpy()[0]

            avg_gain = max(gain, 1e-6)
            avg_loss = max(loss, 1e-6)

            rs = abs(avg_gain / avg_loss) if avg_loss != 0 else float("inf")
            rsi = 100 - (100 / (1 + rs))
            rsi_scores.append(rsi * nifty50_returns[i])

        top_indices = sorted(range(len(rsi_scores)), key=lambda i: rsi_scores[i], reverse=True)[: self._top_n]

        selected_symbols = [symbols[i] for i in top_indices]
        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest