from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy capitalizes on price movements following a significant breakout. "
        "After identifying a strong trend after a price breakouts from support or resistance levels, "
        "it aims to capture the continuation phase of the breakout where price momentum is likely to persist."
    )

    def __init__(self, lookback: int = 60, breakout_margin: float = 0.02, ranking_window: int = 10, max_positions: int = 20) -> None:
        self._lookback = lookback
        self._breakout_margin = breakout_margin
        self._ranking_window = ranking_window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            symbols_history = history.filter(pl.col("symbol") == symbol).sort("session_date")
            closes = symbols_history["adj_close"].to_list()
            high_prices = symbols_history["high"].to_list()
            low_prices = symbols_history["low"].to_list()

            if len(closes) < self._lookback:
                continue

            last_close = float(symbols_history.filter(pl.col("session_date") == view.as_of)["adj_close"].item())
            breakout_price = max(high_prices[-1], last_close * (1 + self._breakout_margin))
            if last_close > high_prices[0] and last_close >= breakout_price:
                breakout_symbols.append(symbol)

        ranked_signals = []
        for symbol in breakout_symbols:
            recent_history = history.filter(pl.col("symbol") == symbol).sort("session_date").tail(self._ranking_window)
            daily_returns = [(float(recent_history.filter(pl.col("session_date") == d["session_date"].item())["adj_close"]) / float(recent_history.filter(pl.col("session_date") == (d["session_date"] - pl.duration(days=1))).select("adj_close").item()) - 1.0) for _, d in recent_history.iter_rows()]
            volume = [float(d["volume"].item()) for _, d in recent_history.iter_rows()]

            avg_return = sum(daily_returns) / len(daily_returns)
            avg_volume = sum(volume) / len(volume)

            ranked_signals.append((symbol, avg_return, avg_volume))

        ranked_signals.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        selected_symbols = [s for s, _, _ in ranked_signals][:self._max_positions]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest