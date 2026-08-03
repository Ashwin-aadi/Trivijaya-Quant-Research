## Cross-Sectional Momentum

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum126(Strategy):
    rationale = (
        "Investors underreact to gradual news, so a stock that has outperformed its peers "
        "over the past six months tends to keep outperforming for several months more. The "
        "most recent week is skipped because short-horizon returns reverse and would "
        "contaminate the medium-term signal. Ranking within the NIFTY 100 rather than "
        "against zero keeps the bet on relative strength instead of on market direction."
    )

    def __init__(self, lookback: int = 126, skip: int = 5, top_n: int = 10) -> None:
        self._lookback = lookback
        self._skip = skip
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        need = self._lookback + self._skip
        closes = view.closes(lookback=need)
        if closes.height < need:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < need:
                continue
            window = values[-need:]
            start = window[0]
            end = window[-1 - self._skip]
            if start <= 0.0 or end <= 0.0:
                continue
            scores[symbol] = end / start - 1.0

        picks = _top_symbols(scores, self._top_n)
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _top_symbols(scores: dict[str, float], top_n: int) -> list[str]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [symbol for symbol, score in ranked[:top_n] if score > 0.0]


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


class ShortHorizonReversal5d(Strategy):
    rationale = (
        "A sharp one-week drop in a large-cap name is usually liquidity demand rather than "
        "news: someone had to sell size quickly and paid for immediacy. Supplying that "
        "liquidity earns the concession back as the order flow is absorbed over the "
        "following days. The effect is strongest in the largest, most heavily traded "
        "stocks, which is exactly what the NIFTY 100 universe contains."
    )

    def __init__(
        self, lookback: int = 5, top_n: int = 10, min_drop: float = 0.02
    ) -> None:
        self._lookback = lookback
        self._top_n = top_n
        self._min_drop = min_drop

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        need = self._lookback + 1
        closes = view.closes(lookback=need)
        if closes.height < need:
            return Signal(information_available_at=stamp, weights={})

        returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < need:
                continue
            start = values[-need]
            end = values[-1]
            if start <= 0.0 or end <= 0.0:
                continue
            returns[symbol] = end / start - 1.0

        picks = _worst_losers(returns, self._top_n, self._min_drop)
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _worst_losers(
    returns: dict[str, float], top_n: int, min_drop: float
) -> list[str]:
    eligible = [(s, r) for s, r in returns.items() if r <= -abs(min_drop)]
    eligible.sort(key=lambda kv: kv[1])
    return [symbol for symbol, _ in eligible[:top_n]]


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


class VolScaledTrend100(Strategy):
    rationale = (
        "A price above its long moving average identifies a name whose drift is currently "
        "positive, and trends of this kind persist longer than a random walk would imply. "
        "Sizing each position by the inverse of its realised volatility means a quiet stock "
        "and a wild one contribute the same amount of risk, rather than letting the noisiest "
        "holdings dominate the portfolio's return. Gross exposure is capped so the book does "
        "not lever up when the whole market goes quiet."
    )

    def __init__(
        self,
        trend_window: int = 100,
        vol_window: int = 20,
        target_vol: float = 0.15,
        max_weight: float = 0.10,
        max_gross: float = 1.0,
    ) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window
        self._target_vol = target_vol
        self._max_weight = max_weight
        self._max_gross = max_gross

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        need = max(self._trend_window, self._vol_window + 1)
        closes = view.closes(lookback=need)
        if closes.height < need:
            return Signal(information_available_at=stamp, weights={})

        raw: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < need:
                continue
            trend = values[-self._trend_window :]
            if values[-1] <= sum(trend) / len(trend):
                continue
            vol = _annual_vol(values[-(self._vol_window + 1) :])
            if vol <= 0.0:
                continue
            raw[symbol] = min(self._target_vol / vol, self._max_weight)

        weights = _cap_gross(raw, self._max_gross)
        return Signal(information_available_at=stamp, weights=weights)


def _annual_vol(values: list[float]) -> float:
    rets: list[float] = []
    for prev, curr in zip(values[:-1], values[1:]):
        if prev > 0.0:
            rets.append(curr / prev - 1.0)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return (var**0.5) * (252.0**0.5)


def _cap_gross(raw: dict[str, float], max_gross: float) -> dict[str, float]:
    total = sum(raw.values())
    if total <= 0.0:
        return {}
    scale = min(1.0, max_gross / total)
    return {symbol: weight * scale for symbol, weight in raw.items()}


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
        "Equal weighting avoids the concentration that market-cap indices carry, but it only "
        "works if every name can actually be traded at the size the book requires. Screening "
        "on median daily traded value keeps the portfolio in stocks where a rebalance costs "
        "basis points rather than percent, and the median is used instead of the mean so a "
        "single block trade cannot promote an otherwise thin stock into the book."
    )

    def __init__(
        self,
        window: int = 60,
        top_n: int = 20,
        min_sessions: int = 40,
        min_turnover: float = 0.0,
    ) -> None:
        self._window = window
        self._top_n = top_n
        self._min_sessions = min_sessions
        self._min_turnover = min_turnover

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        hist = view.history(lookback=self._window)
        if hist.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scores = _turnover_scores(hist, view.symbols, self._min_sessions)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        picks = [
            symbol
            for symbol, value in ranked[: self._top_n]
            if value >= self._min_turnover
        ]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _turnover_scores(
    hist: pl.DataFrame, symbols: tuple[str, ...], min_sessions: int
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for symbol in symbols:
        sub = hist.filter(pl.col("symbol") == symbol)
        if sub.height < min_sessions:
            continue
        prices = [float(v) for v in sub["close"].drop_nulls().to_list()]
        volumes = [float(v) for v in sub["volume"].drop_nulls().to_list()]
        if not prices or len(prices) != len(volumes):
            continue
        traded = [p * v for p, v in zip(prices, volumes) if p > 0.0 and v > 0.0]
        if len(traded) < min_sessions:
            continue
        scores[symbol] = _median(traded)
    return scores


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2
    if n % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


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


class TrailingBandReversion60(Strategy):
    rationale = (
        "A large-cap index constituent has a slow-moving fair value, so its price tends to "
        "oscillate around a trailing average rather than wander away from it. Measuring the "
        "gap in units of the stock's own dispersion makes a two-sigma discount mean the same "
        "thing in a defensive name as in a cyclical one. Buying only the deepest discounts "
        "and holding until the band is regained harvests that oscillation while keeping the "
        "book out of names that are merely drifting."
    )

    def __init__(
        self,
        window: int = 60,
        top_n: int = 8,
        entry_z: float = 1.5,
    ) -> None:
        self._window = window
        self._top_n = top_n
        self._entry_z = entry_z

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        zscores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            score = _distance_z(values[-self._window :])
            if score is None:
                continue
            zscores[symbol] = score

        picks = _deepest_discounts(zscores, self._top_n, self._entry_z)
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _distance_z(window: list[float]) -> float | None:
    reference = sum(window) / len(window)
    if reference <= 0.0:
        return None
    var = sum((p - reference) ** 2 for p in window) / (len(window) - 1)
    dispersion = var**0.5
    if dispersion <= 0.0:
        return None
    return (window[-1] - reference) / dispersion


def _deepest_discounts(
    zscores: dict[str, float], top_n: int, entry_z: float
) -> list[str]:
    eligible = [(s, z) for s, z in zscores.items() if z <= -abs(entry_z)]
    eligible.sort(key=lambda kv: kv[1])
    return [symbol for symbol, _ in eligible[:top_n]]


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```
