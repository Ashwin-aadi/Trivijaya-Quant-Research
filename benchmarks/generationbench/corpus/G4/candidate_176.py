from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMove(Strategy):
    rationale = (
        "This strategy aims to exploit volume-confirmed directional moves in the Indian market. "
        "It identifies breakouts from consolidation patterns or trend lines and confirms them with increased trading volume."
    )

    def __init__(self, window: int = 20, volume_threshold: float = 1.5) -> None:
        self._window = window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            opens = [float(v) for v in history.select(pl.col(symbol).shift(1)).column(symbol)]
            closes = [float(v) for v in history.select(pl.col(symbol).fill_null(pl.lit(0))).column(symbol)]
            high = [float(v) for v in history.select(pl.col(symbol)).column(symbol)]
            low = [float(v) for v in history.select(pl.col(symbol)).column(symbol)]
            volume = [float(v) for v in history.select(pl.col(f"{symbol}_volume")).column(f"{symbol}_volume")]

            if len(opens) < self._window:
                continue

            # Calculate the breakout condition
            breakout_conditions = [
                closes[i] > opens[-1] and (closes[i] - high[i]) > 0.5 * (high[i] - low[i])
                for i in range(len(closes))
            ]
            if any(breakout_conditions):
                recent_breakout_index = [i for i, b in enumerate(breakout_conditions) if b][-1]
                price_move = closes[recent_breakout_index] - opens[-1]

                # Check volume condition
                avg_volume = sum(volume) / len(volume)
                if volume[recent_breakout_index] > self._volume_threshold * avg_volume:
                    picks.append(symbol)

        picks = picks[:20]  # Limit to top 20 symbols
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