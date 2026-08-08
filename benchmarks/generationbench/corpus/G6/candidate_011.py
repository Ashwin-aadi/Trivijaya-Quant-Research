from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "This strategy combines price and volume momentum with earnings growth forecast accuracy "
        "to identify stocks with strong short-term performance and potential for long-term growth. "
        "The dual-momentum approach ensures a balanced portfolio that balances exploration of robust "
        "short-term signals with the consideration of fundamental factors."
    )

    def __init__(self, window_price: int = 20, window_vol: int = 20) -> None:
        self._window_price = window_price
        self._window_vol = window_vol

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_price)
        if closes.height < self._window_price:
            return Signal(information_available_at=stamp, weights={})

        vol_df = view.history(lookback=self._window_vol).sort("session_date")
        if vol_df.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in vol_df["symbol"].to_list():
                continue

            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            volume_values = [
                float(vol_df[(vol_df["symbol"] == symbol)]["volume"]) for _ in range(self._window_vol)
            ]
            if len(close_values) < self._window_price or len(volume_values) != self._window_vol:
                continue

            price_return = (close_values[-1] - close_values[0]) / close_values[0]
            vol_return = volume_values[-1] / volume_values[0]

            # Calculate earnings growth forecast accuracy
            latest_close = view.latest_close()[symbol]
            closes_series = pl.DataFrame({"date": vol_df["session_date"], "close": close_values})
            mean_price_return = (
                closes_series.sort("date")
                .group_by(pl.col("date").dt.year())
                .agg((pl.col("close") / pl.col("close").shift(1) - 1.0).alias("r"))
                .select(pl.col("r").mean().alias("m"))
                .to_series()
            )
            accuracy = abs(mean_price_return[-1] - price_return)

            score = (price_return + vol_return - accuracy) / self._window_price
            if score > 0.5:
                picks.append(symbol)

        picks = picks[:30]
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