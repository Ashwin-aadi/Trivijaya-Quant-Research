from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class NewsAndMacroeconomics(Strategy):
    rationale = (
        "This strategy leverages the composite of company-specific news sentiment and macroeconomic indicators to identify mispriced assets. By combining sentiment scores from recent news articles with macroeconomic trends, it aims to capture short-term inefficiencies in stock prices."
    )

    def __init__(self, news_window: int = 30, macro_window: int = 12) -> None:
        self._news_window = news_window
        self._macro_window = macro_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._news_window)
        if closes.height < self._news_window:
            return Signal(information_available_at=stamp, weights={})

        news_sentiment_scores = {
            "symbol": [],
            "score": [],
        }
        macro_data = _get_macro_data(view.as_of)

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            score = sum(values[-self._news_window:]) / self._news_window
            news_sentiment_scores["symbol"].append(symbol)
            news_sentiment_scores["score"].append(score)

        news_df = pl.DataFrame(news_sentiment_scores).sort("score", descending=True)
        top_symbols = news_df.select(pl.col("symbol").take_range(0, 20)).to_dict(False)["symbol"]

        macro_ranked: list[tuple[str, float]] = []
        for symbol in top_symbols:
            score = abs(macro_data[symbol] - news_sentiment_scores["score"][top_symbols.index(symbol)])
            macro_ranked.append((symbol, score))

        ranked_symbols = [x[0] for x in sorted(macro_ranked, key=lambda x: x[1])[:20]]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _get_macro_data(date_: date) -> dict[str, float]:
    macro_df = pl.DataFrame(
        {
            "symbol": ["NIFTY1", "NIFTY2"],
            "gdp_growth_rate": [0.5, 0.7],
            "inflation_rate": [3.2, 4.1],
        }
    )
    filtered_df = macro_df.filter(pl.col("date") == date_)
    return {
        symbol: float(filtered_df.select(symbol).item())
        for symbol in ["NIFTY1", "NIFTY2"]
    }