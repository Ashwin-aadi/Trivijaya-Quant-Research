from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "This strategy exploits the tendency of market-wide sentiment to influence stock price "
        "movements. During strong market cycles, sectors experience dispersion where individual "
        "stocks deviate from their historical ranges. As market sentiment shifts towards "
        "normalization, these dispersions are compressed, providing an opportunity to profit "
        "from this reversion."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [sym for sym in view.symbols if sym in history.columns]
        hls: list[float] = []
        avg_hls: dict[str, float] = {}
        for symbol in symbols:
            daily_history = history[[symbol, "session_date"]]
            hl_range = (daily_history["high"] - daily_history["low"]).to_list()
            avg_hl = sum(hl_range) / len(hl_range)
            avg_hls[symbol] = avg_hl
            hls.extend([hl / avg_hl for hl in hl_range])

        dispersion = {sym: abs(20 / (1 + 5 * pl.col(sym).mean()) - avg_hl) for sym, avg_hl in avg_hls.items()}
        ranked_dispersion = sorted(dispersion.items(), key=lambda item: item[1], reverse=True)
        picks = [symbol for symbol, _ in ranked_dispersion[:self._top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest