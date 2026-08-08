from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy capitalizes on the relationship between market volatility and trending "
        "behavior in Indian equities. By scaling positions based on realized volatility, it aims to "
        "capture gains during trending periods while mitigating losses by reducing exposure during "
        "calmer markets."
    )

    def __init__(self, window: int = 20, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        log_returns = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window + 1:
                continue

            # Compute daily log returns
            returns = [
                (pl.col(f"{symbol}") / pl.col(f"{symbol}").shift(1)).log()
                .to_numpy()[1:]
                for symbol in view.symbols
            ]
            log_returns.append(
                sum(returns) if len(returns) > 0 else [float("nan")]
            )

        # Flatten the list of returns into a single column
        log_returns = pl.DataFrame({"log_return": [item for sublist in log_returns for item in sublist]})
        realized_volatility = (log_returns.std().item() * (252 ** 0.5)).round(4)
        
        if realized_volatility.is_nan():
            return Signal(information_available_at=stamp, weights={})

        # Rank symbols based on their volatility
        ranked_symbols = log_returns.height - log_returns["log_return"].rank(method="dense", descending=True).to_list()

        picks: list[str] = [view.symbols[i] for i in range(self._top_n) if i < len(view.symbols)]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest