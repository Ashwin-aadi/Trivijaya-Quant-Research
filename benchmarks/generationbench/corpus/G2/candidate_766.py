from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Seasonality in stock markets often arises due to recurring events such as "
        "fiscal years end, earnings seasons, or cultural and environmental factors. "
        "Historically, certain stocks may exhibit higher returns during specific months of the year."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter for NIFTY 100 symbols
        nifty_100_symbols = [s for s in view.symbols if s.startswith("NIFTY")]
        
        seasonality_signals: dict[str, float] = {}
        for symbol in nifty_100_symbols:
            df = history.select(
                pl.col("session_date"), 
                pl.col(symbol)
            ).filter(pl.col("session_date").dt.month().is_in([9, 12]))

            # Calculate the average return for September and December
            if df.height > 1:
                returns = (
                    (df["adj_close"] / df["adj_close"].shift(1) - 1.0).drop_nulls()
                )
                avg_return_september = float(returns.filter(pl.col("session_date").dt.month() == 9).mean())
                avg_return_december = float(returns.filter(pl.col("session_date").dt.month() == 12).mean())

                # If December returns are significantly higher, consider buying in September
                if avg_return_december > avg_return_september:
                    seasonality_signals[symbol] = (avg_return_december - avg_return_september) * 0.5

        # Sort symbols by their potential return and pick the top few
        sorted_symbols = sorted(seasonality_signals.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_symbols[:3]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest