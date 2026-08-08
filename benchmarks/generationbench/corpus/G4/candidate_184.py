from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAverage(Strategy):
    rationale = (
        "This strategy exploits the economic mechanism of price-level reversion against a "
        "trailing reference. It suggests that stock prices tend to revert back to historical "
        "average levels over time, as markets correct deviations from long-term norms."
    )

    def __init__(self, window: int = 200, threshold: float = 1.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            history.groupby("symbol")
                   .agg((pl.col("adj_close").mean()).alias("avg_close"))
                   .collect()
        )
        latest_closes = view.closes(lookback=None)
        deviations: dict[str, float] = {}
        
        for symbol in view.symbols:
            if symbol not in avg_close["symbol"].to_list():
                continue
            avg_close_val = float(avg_close.filter(pl.col("symbol") == symbol)["avg_close"])
            latest_close_val = float(latest_closes[symbol].to_list()[-1])
            deviation = (latest_close_val - avg_close_val) / avg_close_val if avg_close_val != 0 else 0
            deviations[symbol] = deviation

        sorted_symbols = [s for s, d in sorted(deviations.items(), key=lambda item: abs(item[1]))]
        
        buy_symbols = sorted_symbols[:len(sorted_symbols)//2]
        sell_symbols = sorted_symbols[-len(sorted_symbols)//2:]
        
        weights = {symbol: 1.0 / len(buy_symbols) if symbol in buy_symbols else -1.0 / len(sell_symbols) for symbol in view.symbols}
        return Signal(
            information_available_at=stamp, 
            weights={s: w for s, w in weights.items() if w != 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest