from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "This strategy identifies stocks that have deviated significantly from their 52-week "
        "moving average (SMA) and exploits mean reversion. Overvalued stocks are sold, while "
        "undervalued stocks are bought, with positions weighted to balance risk."
    )

    def __init__(self, lookback: int = 52 * 20, threshold_up: float = 1.2, threshold_down: float = 0.8) -> None:
        self._lookback = lookback
        self._threshold_up = threshold_up
        self._threshold_down = threshold_down

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        sma_column_name = f"sma_{self._lookback}w"
        price_level_diffs: list[float] = []
        selected_symbols: set[str] = set()

        for symbol in history["symbol"].to_list():
            if symbol not in closes.columns:
                continue
            session_dates = [date.fromisoformat(d) for d in history[history["symbol"] == symbol]["session_date"]]
            close_values = [float(v) for v in closes[symbol].to_list()]

            # Calculate 52-week SMA
            sma = sum(close_values[-self._lookback:]) / self._lookback

            # Compute deviation from SMA
            recent_close = float(closes[stamp.isoformat()][symbol])
            deviation = (recent_close - sma) / sma if sma != 0 else 0

            price_level_diffs.append(deviation)
            selected_symbols.add(symbol)

        ranked_symbols = [sym for _, sym in sorted(zip(price_level_diffs, selected_symbols), key=lambda x: abs(x[0]), reverse=True)]

        weights: dict[str, float] = {}
        if len(ranked_symbols) > 30:
            top_n = min(20, len(ranked_symbols))
            for symbol in ranked_symbols[:top_n]:
                weights[symbol] = 1.0 / top_n
            return Signal(information_available_at=stamp, weights=weights)

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest