from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are likely to continue in the near term. "
        "This strategy identifies stocks that have recently moved significantly with high volume."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.select("symbol").to_series().unique().to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol)
            opens = [float(o) for o in df["open"].drop_nulls().to_list()]
            closes = [float(c) for c in df["close"].drop_nulls().to_list()]
            volumes = [float(v) for v in df["volume"].drop_nulls().to_list()]

            if len(opens) < self._window or len(closes) < self._window:
                continue

            price_change = (closes[-1] - opens[0]) / opens[0]
            volume_change = volumes[-1]

            # Filter out symbols that do not show significant price change
            if abs(price_change) < 0.05 or volume_change < 1e6:
                continue

            signals[symbol] = float(price_change * volume_change)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted(signals.keys(), key=lambda s: -signals[s])[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().strftime("%Y-%m-%d")
    assert isinstance(newest, str)
    return pl.from_str(newest).date()[0]