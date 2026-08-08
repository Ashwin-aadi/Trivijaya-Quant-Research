from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy leverages the tendency for trending behavior in equity markets during "
        "periods of high volatility. High market volatility often precedes significant price "
        "movements due to increased investor sentiment or news-driven fluctuations. Historically, "
        "such environments can lead to more pronounced trends as investors rush to capitalize on "
        "perceived opportunities or risks."
    )

    def __init__(self, lookback_vol: int = 30, volatility_threshold: float = 2.0, trend_window: int = 10, holding_period: int = 5, max_positions: int = 10) -> None:
        self._lookback_vol = lookback_vol
        self._volatility_threshold = volatility_threshold
        self._trend_window = trend_window
        self._holding_period = holding_period
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_vol)
        if closes.height < self._lookback_vol:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        daily_returns = (closes.to_pandas() / closes.shift(1).to_pandas().fillna(1) - 1).dropna()

        # Calculate 20-day standard deviation of returns
        std_dev_20 = daily_returns.rolling(window=20).std().iloc[-1]

        if std_dev_20 < self._volatility_threshold:
            return Signal(information_available_at=stamp, weights={})

        # Identify trending symbols by comparing SMAs
        sma_upward = daily_returns.rolling(window=self._trend_window).mean().shift(-self._trend_window + 1)
        sma_downward = -daily_returns.rolling(window=self._trend_window).mean().shift(-self._trend_window + 1)

        upward_changes = (sma_upward > sma_upward.shift(1)).astype(int)
        downward_changes = (sma_downward < sma_downward.shift(1)).astype(int)

        # Rank symbols by trend strength
        ranked_symbols = []
        for symbol in view.symbols:
            if symbol not in daily_returns.columns or symbol not in sma_upward.columns:
                continue

            upward_strength = float(upward_changes[symbol].sum())
            downward_strength = float(downward_changes[symbol].sum())

            score = upward_strength - downward_strength
            ranked_symbols.append((symbol, score))

        ranked_symbols.sort(key=lambda x: x[1], reverse=True)

        if len(ranked_symbols) < self._max_positions:
            return Signal(information_available_at=stamp, weights={})

        # Allocate weights based on trend strength
        top_symbols = [symbol for symbol, _ in ranked_symbols[:self._max_positions]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest