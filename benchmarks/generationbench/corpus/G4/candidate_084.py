from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumUndervaluation(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by identifying stocks that have "
        "performed well recently but are currently undervalued. It ranks stocks based on their "
        "short-term returns and screens for those with negative valuation metrics relative to "
        "the market average, aiming to capitalize on both the continuation of past performance "
        "and potential undervaluation."
    )

    def __init__(self, momentum_window: int = 30, valuation_window: int = 250, top_n: int = 30) -> None:
        self._momentum_window = momentum_window
        self._valuation_window = valuation_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window)
        if closes.height < self._momentum_window:
            return Signal(information_available_at=stamp, weights={})

        market_average = view.history().select(
            pl.col("adj_close").mean().alias("market_avg")
        ).collect().row(0)[0]

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._momentum_window:
                continue
            momentum = (values[-1] - values[0]) / values[0]
            if values[-1] > market_average and momentum >= 0.05:  # Adjust threshold as needed
                valuation_history = view.history().select(
                    pl.col("symbol"), pl.col("adj_close") / pl.col("adj_close").shift(self._valuation_window).alias("p_e")
                ).collect()
                if symbol in valuation_history["symbol"].to_list():
                    pe_ratio = float(valuation_history.filter(pl.col("symbol") == symbol)["p_e"].item())
                    if pe_ratio < market_average * 0.8:  # Adjust threshold as needed
                        picks.append(symbol)

        picks = picks[: self._top_n]
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