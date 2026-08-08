from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Identifying stocks with extreme price deviations from their historical averages "
        "can capitalize on short-horizon mean reversion. This strategy aims to profit by "
        "short selling overvalued stocks and buying undervalued ones."
    )

    def __init__(self, window: int = 20, threshold: float = 5.0, top_n: int = 40) -> None:
        self._window = window
        self._threshold = threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate 20-day simple moving average (SMA)
        sma = history.groupby("symbol").agg(
            (pl.col("adj_close").mean().alias("sma"))
        )

        # Merge SMA with the full history to compute daily percentage deviation
        merged = history.join(sma, on="symbol", how="inner")
        merged = merged.with_columns(
            (100 * (pl.col("adj_close") - pl.col("sma")) / pl.col("sma")).alias("deviation_percentage")
        )

        # Filter stocks with deviations exceeding the threshold
        filtered = merged.filter(
            (pl.col("deviation_percentage").abs() > self._threshold)
        ).sort("deviation_percentage", descending=True)

        if filtered.height < 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in filtered["symbol"].to_list():
                continue
            deviation = float(filtered[filtered["symbol"] == symbol]["deviation_percentage"])
            if abs(deviation) > self._threshold:
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: -weight for s in picks[:20]}, cash_weight=1 + sum(weight * 20)
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest