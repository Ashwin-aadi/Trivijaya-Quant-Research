from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "This strategy leverages the combination of earnings surprises and macroeconomic "
        "indicators to identify undervalued stocks. Earnings surprises signal unexpected "
        "performance, while macroeconomic signals provide a context for broader market "
        "conditions."
    )

    def __init__(self, window_earnings: int = 3, window_macro: int = 6) -> None:
        self._window_earnings = window_earnings
        self._window_macro = window_macro

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_earnings + self._window_macro)

        if closes.height < self._window_earnings + self._window_macro:
            return Signal(information_available_at=stamp, weights={})

        macro_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in macro_signals:
                macro_signals[symbol] = 0.0

        earnings_surprises: dict[str, float] = {}
        for symbol in view.symbols:
            latest_close = float(view.latest_close()[symbol])
            history = closes[[symbol, "session_date"]]
            history = history.sort("session_date")
            history = history.with_columns(
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).alias("return")
            )
            returns = [float(v) for v in history["return"].to_list()]
            mean_return = sum(returns[-self._window_earnings:]) / self._window_earnings

            eps_reported = float(closes[[symbol, "session_date"]].filter(
                pl.col("session_date") == stamp
            )["adj_close"])
            expected_eps = (closes[[symbol, "session_date"]].head(1)["adj_close"]
                            if closes.filter(pl.col("symbol") == symbol).height > 0 else 0)
            earnings_surprise = eps_reported - expected_eps

            macro_signal = 0.0
            for m_symbol in view.symbols:
                recent_history = view.history(lookback=self._window_macro)[[m_symbol, "session_date"]]
                recent_history = recent_history.sort("session_date")
                macro_returns = [float(v) for v in recent_history["adj_close"].to_list()]
                mean_macro_return = sum(macro_returns[-self._window_macro:]) / self._window_macro
                if m_symbol == symbol:
                    continue
                if abs(mean_macro_return - mean_return) < 0.01:
                    macro_signal += 0.1

            earnings_surprises[symbol] = earnings_surprise
            macro_signals[symbol] = macro_signal

        weighted_scores: dict[str, float] = {}
        for symbol in view.symbols:
            combined_score = (earnings_surprises[symbol] + macro_signals[symbol]) / 2.0
            if combined_score > 0:
                weighted_scores[symbol] = combined_score

        sorted_symbols = sorted(weighted_scores.items(), key=lambda x: -x[1])
        top_stocks = [s for s, _ in sorted_symbols[:5]]

        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.2 / len(top_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest