from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the strongest relative performance against a broad index "
        "can provide an edge in equity markets. This strategy focuses on stocks that have "
        "outperformed the market over the past 30 days."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_100 = [symbol for symbol in view.symbols if "NIFTY" not in symbol]
        broad_index = ["^NSEI"]  # Assuming ^NSEI as the NIFTY index
        nifty_100 += broad_index

        closes = history.to_pandas().set_index("session_date")
        returns = (closes["close"].pct_change() - closes[broad_index[0]]["close"].pct_change()).dropna()

        top_n = 5
        top_stocks = returns.nlargest(top_n).index.get_level_values(1).tolist()
        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest