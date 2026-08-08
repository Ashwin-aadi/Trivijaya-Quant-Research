from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalVolatility(Strategy):
    rationale = (
        "This strategy leverages the combination of seasonality and recent volatility to identify "
        "potentially significant price movements. High seasonality suggests structured patterns, while moderate"
        "volatility indicates heightened trading activity that could amplify these patterns."
    )

    def __init__(self, window_season: int = 30, window_volatility: int = 30, top_n: int = 20) -> None:
        self._window_season = window_season
        self._window_volatility = window_volatility
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_season + self._window_volatility)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        seasonality_signal = self._calculate_seasonality_signal(history)
        volatility_signal = self._calculate_volatility_signal(history)

        if not seasonality_signal or not volatility_signal:
            return Signal(information_available_at=stamp, weights={})

        scores = [
            (seasonality_score + volatility_score) / 2 for
            seasonality_score, volatility_score in zip(seasonality_signal, volatility_signal)
        ]

        ranked_stocks = self._rank_stocks(scores, view.symbols)

        if not ranked_stocks:
            return Signal(information_available_at=stamp, weights={})

        top_n_stocks = ranked_stocks[:self._top_n]
        weight = 1.0 / len(top_n_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_stocks}
        )

    def _calculate_seasonality_signal(self, history: pl.DataFrame) -> list[float]:
        seasonal_dates = self._get_seasonal_dates()
        seasonality_scores = []

        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            recent_prices = [float(v) for v in df["adj_close"].to_list()[-self._window_season:]]
            seasonal_price = sum([recent_prices[date] for date in seasonal_dates if date in range(df.height)])
            seasonality_scores.append(seasonal_price)

        return seasonality_scores

    def _calculate_volatility_signal(self, history: pl.DataFrame) -> list[float]:
        volatility_scores = []

        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            recent_prices = [float(v) for v in df["adj_close"].to_list()[-self._window_volatility:]]
            std_deviation = pl.Series(recent_prices).std()
            volatility_scores.append(std_deviation)

        return volatility_scores

    def _rank_stocks(self, scores: list[float], symbols: tuple[str, ...]) -> list[str]:
        ranked_stocks = sorted(zip(scores, symbols), key=lambda x: -x[0])
        return [s for _, s in ranked_stocks]

    def _get_seasonal_dates(self) -> set[int]:
        # Placeholder logic to get seasonal dates
        current_date = view.as_of
        festival_weeks = {15, 20}  # Example week numbers during a festival period
        return {current_date.isocalendar().week for w in festival_weeks}

def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest