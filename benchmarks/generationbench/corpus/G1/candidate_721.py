from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion suggests that asset prices will revert to their historical "
        "mean. By identifying stocks whose recent prices deviate significantly from their moving "
        "average, we can exploit this tendency for profitable trading opportunities."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        symbol_close_mean = closes.select(
            pl.col(symbols).mean().alias("mean")
        ).to_dict()[0]["mean"]
        
        symbols_to_trade: list[str] = []
        for symbol in symbols:
            if symbol not in closes.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            z_score = (recent_closes[-1] - symbol_close_mean) / pl.col(symbols).mean().std()
            if abs(z_score) > 2.0:  # Consider using a different threshold
                symbols_to_trade.append(symbol)

        weights = {symbol: 1.0 / len(symbols_to_trade) for symbol in symbols_to_trade}
        return Signal(
            information_available_at=stamp, weights={s: weights[s] for s in symbols_to_trade}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest