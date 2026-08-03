## Cross-Sectional Momentum

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Investors underreact to gradually diffusing information, so stocks that have already "
        "outperformed their peers over the past six months tend to keep doing so over the next "
        "few weeks. The most recent month is skipped because short-horizon returns are "
        "contaminated by bid-ask bounce and liquidity-driven reversal, which works against the "
        "momentum effect. Ranking across the cross-section rather than against a stock's own "
        "history keeps the book neutral to broad market direction."
    )

    def __init__(
        self, formation: int = 126, skip: int = 21, top_n: int = 10
    ) -> None:
        self._formation = formation
        self._skip = skip
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        window = self._formation + self._skip
        closes = view.closes(lookback=window)
        if closes.height < window:
            return Signal(information_available_at=stamp, weights={})

        scores: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < window:
                continue
            start = values[-window]
            end = values[-(self._skip + 1)]
            if start <= 0.0:
                continue
            scores.append((symbol, end / start - 1.0))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        picks = [symbol for symbol, score in scores[: self._top_n] if score > 0.0]
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
```

## Short-Horizon Mean Reversion

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ShortHorizonReversal(Strategy):
    rationale = (
        "Over a handful of sessions price moves are dominated by liquidity demand rather than "
        "news: an impatient seller pushes a stock below fair value and is compensated by whoever "
        "takes the other side. Buying the worst performers of the last week is a way of supplying "
        "that liquidity and collecting the rebate when the pressure abates. The effect is "
        "strongest in large, heavily traded names where the flow is transient rather than "
        "informed, which is exactly the NIFTY 100 universe."
    )

    def __init__(
        self, lookback: int = 5, bottom_n: int = 10, min_drop: float = 0.01
    ) -> None:
        self._lookback = lookback
        self._bottom_n = bottom_n
        self._min_drop = min_drop

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        window = self._lookback + 1
        closes = view.closes(lookback=window)
        if closes.height < window:
            return Signal(information_available_at=stamp, weights={})

        scores = self._score(view, closes, window)
        scores.sort(key=lambda pair: pair[1])
        picks = [
            symbol for symbol, score in scores[: self._bottom_n]
            if score <= -self._min_drop
        ]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )

    def _score(
        self, view: MarketView, closes: pl.DataFrame, window: int
    ) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < window:
                continue
            start = values[-window]
            if start <= 0.0:
                continue
            scores.append((symbol, values[-1] / start - 1.0))
        return scores


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```

## Volatility-Scaled Trend Following

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolScaledTrend(Strategy):
    rationale = (
        "A price sitting above its long moving average is the crudest and most durable evidence "
        "that a trend is intact, and trends persist because capital reallocates into winners "
        "slowly. Sizing each position at a fixed volatility budget rather than equally stops a "
        "few high-beta names from dictating the portfolio's risk, so every holding contributes "
        "a comparable amount of variance. The result is a book whose realised volatility is far "
        "more stable across calm and turbulent regimes than a naive equal-weighted trend book."
    )

    def __init__(
        self,
        trend_window: int = 100,
        vol_window: int = 20,
        target_vol: float = 0.20,
        max_weight: float = 0.15,
        gross_cap: float = 1.0,
    ) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window
        self._target_vol = target_vol
        self._max_weight = max_weight
        self._gross_cap = gross_cap

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        window = max(self._trend_window, self._vol_window) + 1
        closes = view.closes(lookback=window)
        if closes.height < window:
            return Signal(information_available_at=stamp, weights={})

        raw: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < window:
                continue
            trend = values[-self._trend_window :]
            if values[-1] <= sum(trend) / len(trend):
                continue
            vol = _annualised_vol(values[-(self._vol_window + 1) :])
            if vol <= 0.0:
                continue
            raw[symbol] = min(self._target_vol / vol, self._max_weight)

        return _scaled_signal(stamp, raw, self._gross_cap)


def _annualised_vol(values: list[float]) -> float:
    rets = [
        values[i] / values[i - 1] - 1.0
        for i in range(1, len(values))
        if values[i - 1] > 0.0
    ]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return (var ** 0.5) * (252.0 ** 0.5)


def _scaled_signal(
    stamp: date, raw: dict[str, float], gross_cap: float
) -> Signal:
    if not raw:
        return Signal(information_available_at=stamp, weights={})
    total = sum(raw.values())
    if total > gross_cap and total > 0.0:
        scale = gross_cap / total
        raw = {symbol: w * scale for symbol, w in raw.items()}
    return Signal(information_available_at=stamp, weights=raw)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```

## Liquidity-Screened Equal Weighting

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeight(Strategy):
    rationale = (
        "Equal weighting is a strong default because it avoids concentrating the book in whatever "
        "the index happens to have bid up, but it is only implementable in names that can absorb "
        "the trade. Screening on median daily traded value rather than a single day's turnover "
        "removes stocks whose apparent liquidity comes from one block print. Median turnover also "
        "proxies for investor attention and index membership stability, so the surviving names "
        "carry lower transaction costs and less risk of a stale or manipulated print."
    )

    def __init__(
        self, window: int = 60, top_n: int = 20, gross: float = 1.0
    ) -> None:
        self._window = window
        self._top_n = top_n
        self._gross = gross

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        hist = view.history(lookback=self._window)
        if hist.is_empty() or hist.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        turnover = _turnover_by_symbol(hist)
        scores: list[tuple[str, float]] = []
        for symbol in view.symbols:
            series = turnover.get(symbol, [])
            if len(series) < self._window // 2:
                continue
            scores.append((symbol, _median(series)))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        picks = [symbol for symbol, value in scores[: self._top_n] if value > 0.0]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = self._gross / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _turnover_by_symbol(hist: pl.DataFrame) -> dict[str, list[float]]:
    symbols = hist["symbol"].to_list()
    closes = hist["close"].to_list()
    volumes = hist["volume"].to_list()
    out: dict[str, list[float]] = {}
    for symbol, close, volume in zip(symbols, closes, volumes):
        if symbol is None or close is None or volume is None:
            continue
        out.setdefault(str(symbol), []).append(float(close) * float(volume))
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```

## Price-Level Reversion Against a Trailing Reference

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingAnchorReversion(Strategy):
    rationale = (
        "A trailing average price acts as a reference point that market participants anchor on, "
        "so a stock trading well below it looks cheap to the same investors who were happy to own "
        "it a quarter ago. Unlike a few-day reversal signal, the gap to a long anchor measures "
        "displacement in price level rather than recent return, and it closes over weeks rather "
        "than days. A floor on the gap avoids buying names in genuine structural decline, where "
        "the anchor is stale rather than informative."
    )

    def __init__(
        self,
        anchor_window: int = 60,
        top_n: int = 8,
        min_gap: float = 0.05,
        max_gap: float = 0.35,
    ) -> None:
        self._anchor_window = anchor_window
        self._top_n = top_n
        self._min_gap = min_gap
        self._max_gap = max_gap

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._anchor_window)
        if closes.height < self._anchor_window:
            return Signal(information_available_at=stamp, weights={})

        gaps = self._gaps(view, closes)
        gaps.sort(key=lambda pair: pair[1])
        picks = [symbol for symbol, gap in gaps[: self._top_n]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )

    def _gaps(
        self, view: MarketView, closes: pl.DataFrame
    ) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._anchor_window:
                continue
            anchor = sum(values) / len(values)
            if anchor <= 0.0:
                continue
            gap = values[-1] / anchor - 1.0
            if -self._max_gap <= gap <= -self._min_gap:
                out.append((symbol, gap))
        return out


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```
