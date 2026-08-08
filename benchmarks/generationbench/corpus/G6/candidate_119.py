from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class IntegratedStrategy(Strategy):
    rationale = (
        "This strategy combines momentum, sentiment, earnings surprises, and value metrics "
        "to identify high-quality stocks in the Indian market. It aims for a balanced portfolio "
        "by leveraging multiple indicators to ensure robustness and diversification."
    )

    def __init__(self, window: int = 52, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        msi = _market_sentiment_indicator(view)
        vo = _volume_oscillator(closes)
        es = _earnings_surprises(view)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in msi.columns or symbol not in vo.columns or symbol not in es.columns:
                continue
            momentum = (closes[symbol].to_list()[-1] - closes[symbol].to_list()[0]) / \
                       min(closes[symbol].to_list())
            score = 0.3 * _momentum_score(momentum, self._window) + \
                    0.25 * msi[symbol][-1] + \
                    0.25 * vo[symbol][-1] + \
                    0.2 * es[symbol]

            if score >= 0.7:
                picks.append(symbol)

        picks = picks[: self._top_n]
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
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _market_sentiment_indicator(view: MarketView) -> pl.DataFrame:
    history = view.history(lookback=52)
    msi = (history["adj_close"] / history["adj_close"].shift(1)) - 1.0
    return history.select(pl.col("symbol"), msi.sum().alias("msi"))


def _volume_oscillator(closes: pl.DataFrame) -> dict[str, float]:
    short_term = closes.sort("session_date").tail(20)["volume"].mean()
    long_term = closes["volume"].mean()
    return {s: (short_term[s] / long_term - 1.0).item() for s in closes.columns}


def _earnings_surprises(view: MarketView) -> pl.DataFrame:
    history = view.history(lookback=52)
    es = history.select(
        pl.col("symbol"),
        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("es")
    )
    return es.filter(pl.col("es").is_not_null()).group_by("symbol").agg(
        pl.col("es").mean().alias("average_es")
    )


def _momentum_score(momentum: float, window: int) -> float:
    if momentum > 0.1 * (window / 52):
        return 1.0
    elif momentum > -0.1 * (window / 52):
        return 0.5
    else:
        return 0.0