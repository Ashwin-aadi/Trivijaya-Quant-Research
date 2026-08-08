from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ConservativeUnifiedDesign(Strategy):
    rationale = (
        "This strategy evaluates stocks based on their daily price momentum and volume volatility. "
        "Stocks with high momentum and low volume volatility are selected for the portfolio, while a"
        " stop-loss mechanism ensures risk management."
    )

    def __init__(self, window_5_day_high: int = 5, window_20_day_volume: int = 20, top_n_momentum: int = 30, bottom_n_volatility: int = 40) -> None:
        self._window_5_day_high = window_5_day_high
        self._window_20_day_volume = window_20_day_volume
        self._top_n_momentum = top_n_momentum
        self._bottom_n_volatility = bottom_n_volatility

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_5_day_high + self._window_20_day_volume)

        if history.height < self._window_5_day_high + self._window_20_day_volume:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volume_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            close = [float(v) for v in history[symbol]["close"].drop_nulls().to_list()]
            open_ = [float(v) for v in history[symbol]["open"].drop_nulls().to_list()]
            high = [float(v) for v in history[symbol]["high"].drop_nulls().to_list()]
            low = [float(v) for v in history[symbol]["low"].drop_nulls().to_list()]
            volume = [float(v) for v in history[symbol]["volume"].drop_nulls().to_list()]

            # Calculate 5-day high
            five_day_highs = sorted(high[-self._window_5_day_high:], reverse=True)[: self._window_5_day_high]
            last_five_day_high = max(five_day_highs)

            # Calculate momentum score
            if last_five_day_high == 0:
                continue
            momentum_score = ((close[-1] - last_five_day_high) / last_five_day_high * 100)
            momentum_scores[symbol] = momentum_score

            # Calculate 20-day average volume
            twenty_day_volumes = [v for v in volume[-self._window_20_day_volume:]]
            if len(twenty_day_volumes) < self._window_20_day_volume:
                continue
            avg_volume = sum(twenty_day_volumes) / self._window_20_day_volume

            # Calculate volume volatility score
            normalized_volatility = (volume[-1] - avg_volume) / view.latest_close()[symbol]
            if symbol not in history.columns or len(volume) < self._window_20_day_volume:
                continue
            volume_scores[symbol] = normalized_volatility

        top_momentum_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:self._top_n_momentum]
        bottom_volatility_symbols = sorted(volume_scores.items(), key=lambda x: abs(x[1]), reverse=False)[:self._bottom_n_volatility]

        selected_symbols = set()
        for symbol, _ in top_momentum_symbols:
            if symbol not in bottom_volatility_symbols and symbol in history.columns:
                selected_symbols.add(symbol)

        if len(selected_symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest