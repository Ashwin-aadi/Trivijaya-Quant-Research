"""Rebuild a price panel along a resampled date sequence, so strategies re-decide under it.

:mod:`src.stress.crr` returns index sequences and says nothing about what they are applied to,
because that choice is a methodological fork. This module is the PI's answer to it, approved
2026-08-01, and it implements exactly three rules:

1. **Resample returns, never price levels.** Gathering price rows by date index directly would
   splice one period's price level onto another's at every block seam. Over the development window
   ``adj_close`` grows by a median of 2.55x, a 90th percentile of 6.75x and a maximum of 32.75x,
   and a path contains roughly 112 seams at the calibrated block length of 11.07 sessions — so
   level-splicing would manufacture around a hundred fabricated overnight moves per path, some of
   several thousand percent. Those would dominate every fragility number computed downstream. Each
   symbol's synthetic close is therefore compounded from its own real starting price.

2. **Reconstruct the bar by scaling.** The synthetic close implies a ratio against the sampled
   day's real close; open, high and low are multiplied by that same ratio, so the intraday geometry
   of every synthetic bar is the real geometry of some real day. Volume and traded value are
   carried unchanged: they describe the market condition on the sampled day, and that condition is
   what a counterfactual path is meant to transplant.

3. **Universe alignment follows the synthetic timeline, not the sampled day.** The output keeps the
   real development ``session_date`` axis, so the calendar and the point-in-time universe are the
   untouched real objects and the rebalance schedule is the real one. No artificial turnover is
   created at block seams — which matters because cost drag is the dominant effect in this lab's
   results, and seam turnover would contaminate fragility with a cost artefact. The price of this
   choice is that a universe member can have no row on a synthetic day, since 44 of the 185
   symbols do not span the whole window. That rate is measured per path and reported rather than
   assumed small: see :class:`PanelDiagnostics`.

What this deliberately does not do: it never invents a return. A symbol's move into a sampled day is
the move it really made into that day, measured from its **previous available close** rather than
from the previous session — which is what the symbol's own return series says, and what makes a
listing gap a gap rather than a hole. On the one day per symbol that has no previous close at all,
its first ever quotation, the level is carried forward unchanged and no return is claimed.

Measuring returns over sessions rather than over available closes was a real defect in the first
version of this module: it discarded each symbol's first quoted day, 48 rows of 211,927, and on an
identity path — where the resampling is a no-op and the output must equal the input — that moved
one strategy's Sharpe by 0.165. ``tests/stress/test_panel.py`` now pins the identity property.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from src.common.exceptions import DataIntegrityError

#: Columns multiplied by the reconstructed-close ratio. Every one of them is a price, and all of
#: them are raw except the three adjusted ones — which is why ``prev_close`` is scaled here rather
#: than set to the previous synthetic close. It is a raw exchange field, so writing an adjusted
#: level into it disagreed with the panel by a factor of 1.23 across a split. Scaled, it keeps the
#: sampled day's real previous-close-to-open gap, exactly as open, high and low keep that day's
#: real intraday geometry.
_SCALED = ("open", "high", "low", "close", "prev_close", "adj_open", "adj_high", "adj_low")
#: Columns carried verbatim from the sampled day. Volume and traded value are the market condition
#: being transplanted; the divisor is a corporate-action bookkeeping term with no price dimension.
_CARRIED = ("volume", "adj_volume", "turnover_inr", "divisor")


@dataclass(frozen=True)
class PanelDiagnostics:
    """Per-path evidence that the reconstruction did what it claims, reported not asserted."""

    n_sessions: int
    n_symbols: int
    n_rows: int
    #: Universe member-days with no price row, synthetic and — for comparison — real.
    universe_member_days: int
    missing_member_days: int
    missing_member_rate: float
    real_missing_member_rate: float
    #: Rows emitted at a carried-forward level because the sampled day supplied no return — only a
    #: symbol's first ever quotation can do that, so this is bounded by the symbol count.
    carried_forward_days: int
    #: Largest single-session move in the reconstructed panel. A level-splicing bug shows up here
    #: as a multi-hundred-percent move long before it shows up in a Sharpe ratio.
    max_abs_session_return: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_sessions": self.n_sessions,
            "n_symbols": self.n_symbols,
            "n_rows": self.n_rows,
            "universe_member_days": self.universe_member_days,
            "missing_member_days": self.missing_member_days,
            "missing_member_rate": self.missing_member_rate,
            "real_missing_member_rate": self.real_missing_member_rate,
            "carried_forward_days": self.carried_forward_days,
            "max_abs_session_return": self.max_abs_session_return,
        }


class SyntheticPanelBuilder:
    """Turns one bootstrap index path into a panel the engine can run unmodified.

    The dense per-symbol matrices are built once and reused for every path, because that work is
    the expensive part and it does not depend on the path. Constructing this object costs a pass
    over the panel; :meth:`build` is then array arithmetic.
    """

    def __init__(self, panel: pl.DataFrame, universe: pl.DataFrame | None = None) -> None:
        self._schema = panel.schema
        self._sessions: list[date] = sorted(panel["session_date"].unique().to_list())
        self._symbols: list[str] = sorted(panel["symbol"].unique().to_list())
        self._n_sessions = len(self._sessions)
        self._n_symbols = len(self._symbols)
        if self._n_sessions < 2:
            raise DataIntegrityError(
                f"need at least 2 sessions to resample, got {self._n_sessions}"
            )

        self._grid = self._densify(panel)
        close = self._grid["adj_close"]
        self._present = np.isfinite(close)
        self._step = self._log_steps(close)
        self._anchor = self._first_present(close)
        self._universe_mask = self._build_universe_mask(universe)

    # --- one-off preparation ----------------------------------------------------

    def _densify(self, panel: pl.DataFrame) -> dict[str, np.ndarray]:
        """Scatter the long panel into ``(session, symbol)`` matrices, NaN where a row is absent."""
        session_of = {day: i for i, day in enumerate(self._sessions)}
        symbol_of = {sym: i for i, sym in enumerate(self._symbols)}
        rows = np.array([session_of[d] for d in panel["session_date"].to_list()])
        cols = np.array([symbol_of[s] for s in panel["symbol"].to_list()])

        grid: dict[str, np.ndarray] = {}
        for column in (*_SCALED, *_CARRIED, "adj_close", "prev_close"):
            if column not in panel.columns:
                continue
            values = panel[column].cast(pl.Float64).to_numpy()
            matrix = np.full((self._n_sessions, self._n_symbols), np.nan)
            matrix[rows, cols] = values
            grid[column] = matrix
        return grid

    @staticmethod
    def _log_steps(close: np.ndarray) -> np.ndarray:
        """Log return **into** each session from that symbol's previous available close.

        Measured over the symbol's own quotations, so a listing gap is spanned by one real move
        rather than turning into an undefined one. NaN where the symbol did not trade, and on its
        first ever quotation, which has nothing to be measured from.
        """
        step = np.full(close.shape, np.nan)
        for column in range(close.shape[1]):
            quoted = np.flatnonzero(np.isfinite(close[:, column]))
            if quoted.size > 1:
                prices = close[quoted, column]
                step[quoted[1:], column] = np.log(prices[1:] / prices[:-1])
        return step

    @staticmethod
    def _first_present(close: np.ndarray) -> np.ndarray:
        """Each symbol's first real adjusted close: the price its synthetic path compounds from."""
        anchor = np.full(close.shape[1], np.nan)
        for column in range(close.shape[1]):
            finite = np.flatnonzero(np.isfinite(close[:, column]))
            if finite.size:
                anchor[column] = close[finite[0], column]
        return anchor

    def _build_universe_mask(self, universe: pl.DataFrame | None) -> np.ndarray | None:
        """``(session, symbol)`` membership under the real point-in-time universe.

        Mirrors ``BacktestEngine._universe_on``: the effective snapshot on a session is the most
        recent rebalance at or before it. Used only to measure the missing-member rate.
        """
        if universe is None:
            return None
        symbol_of = {sym: i for i, sym in enumerate(self._symbols)}
        mask = np.zeros((self._n_sessions, self._n_symbols), dtype=bool)
        rebalances = sorted(universe["rebalance_date"].unique().to_list())
        members = {
            day: [symbol_of[s] for s in universe.filter(pl.col("rebalance_date") == day)["symbol"]
                  if s in symbol_of]
            for day in rebalances
        }
        for position, day in enumerate(self._sessions):
            effective = [r for r in rebalances if r <= day]
            if effective:
                mask[position, members[effective[-1]]] = True
        return mask

    # --- per-path construction --------------------------------------------------

    def build(self, index_path: np.ndarray) -> tuple[pl.DataFrame, PanelDiagnostics]:
        """Reconstruct the panel along one resampled sequence of return slots.

        ``index_path`` holds ``n_sessions - 1`` positions into the return series, exactly what
        :func:`src.stress.crr.conditional_bootstrap_indices` produces for that length. Synthetic
        session ``j`` carries the move that really occurred into source session ``index_path[j-1]
        + 1``, and session 0 is the real first session verbatim.
        """
        index_path = np.asarray(index_path, dtype=np.int64).ravel()
        expected = self._n_sessions - 1
        if index_path.shape[0] != expected:
            raise DataIntegrityError(
                f"index path has {index_path.shape[0]} steps; this panel needs {expected}"
            )

        source = np.empty(self._n_sessions, dtype=np.int64)
        source[0] = 0
        source[1:] = index_path + 1

        # A symbol has a row on a synthetic session exactly when it had one on the sampled day, so
        # the output's presence pattern is a permutation of the real one rather than a subset.
        present = self._present[source]
        return self._materialise(source, self._compound(source), present)

    def _compound(self, source: np.ndarray) -> np.ndarray:
        """Synthetic adjusted close per session and symbol, compounded from each symbol's anchor.

        A sampled day carrying no return for a symbol — only its first ever quotation does —
        contributes zero to the cumulative sum, which carries the level forward unchanged rather
        than moving it. On an identity path the cumulative sum before a symbol's first quotation is
        empty, so its level there is its anchor: the real close, exactly.
        """
        steps = self._step[source]
        contribution = np.where(np.isfinite(steps), steps, 0.0)
        level: np.ndarray = self._anchor[None, :] * np.exp(np.cumsum(contribution, axis=0))
        return level

    def _materialise(
        self, source: np.ndarray, level: np.ndarray, present: np.ndarray
    ) -> tuple[pl.DataFrame, PanelDiagnostics]:
        """Assemble the long frame and the diagnostics for one path."""
        source_close = self._grid["adj_close"][source]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = level / source_close
        scale[~present] = np.nan

        columns: dict[str, np.ndarray] = {"adj_close": level}
        for name in _SCALED:
            if name in self._grid:
                columns[name] = self._grid[name][source] * scale
        for name in _CARRIED:
            if name in self._grid:
                columns[name] = self._grid[name][source]
        frame = self._to_frame(columns, present)
        carried = int((present & ~np.isfinite(self._step[source])).sum())
        return frame, self._diagnose(level, present, frame.height, carried)

    def _to_frame(self, columns: dict[str, np.ndarray], present: np.ndarray) -> pl.DataFrame:
        """Flatten the matrices to the engine's long schema, session-sorted as PanelIndex requires.

        Row-major order over ``(session, symbol)`` is already sorted by session, so no sort is
        needed and none is done — a sort here would be the single most expensive step per path.

        The ``set_sorted`` call is not cosmetic and not an optimisation. Polars records sortedness
        as column metadata, and the real panel acquires that flag from the ``.sort("session_date")``
        its loader ends with. A frame that is sorted but unflagged is *not* interchangeable with one
        that is flagged: a strategy calling ``.sort("session_date", descending=True)`` gets a cheap
        reversal on a flagged column and a genuine sort on an unflagged one, and those disagree on
        the order of tied rows. ``candidate_072`` does exactly that, and the two panels handed it
        disjoint portfolios — the alphabetically last five symbols against the alphabetically first
        five — from inputs agreeing to 7e-16. Faithfulness here means matching the real panel's
        metadata, not only its values.
        """
        flat = present.ravel()
        session_column = np.repeat(np.arange(self._n_sessions), self._n_symbols)[flat]
        symbol_column = np.tile(np.arange(self._n_symbols), self._n_sessions)[flat]
        sessions = np.array(self._sessions, dtype="datetime64[D]")
        symbols = np.array(self._symbols, dtype=object)

        data: dict[str, pl.Series] = {
            "session_date": pl.Series(sessions[session_column]).cast(pl.Date),
            "symbol": pl.Series(symbols[symbol_column], dtype=pl.String),
        }
        for name, matrix in columns.items():
            data[name] = pl.Series(name, matrix.ravel()[flat]).cast(self._schema[name])
        return (
            pl.DataFrame(data)
            .select([c for c in self._schema.names() if c in data])
            .set_sorted("session_date")
        )

    def _diagnose(
        self, level: np.ndarray, present: np.ndarray, n_rows: int, carried: int
    ) -> PanelDiagnostics:
        """Measure the two things that would be invisible in a Sharpe ratio if they went wrong."""
        with np.errstate(divide="ignore", invalid="ignore"):
            moves = level[1:] / level[:-1] - 1.0
        moves = np.where(present[1:] & present[:-1], moves, np.nan)
        largest = float(np.nanmax(np.abs(moves))) if np.isfinite(moves).any() else 0.0

        member_days = missing_days = 0
        rate = real_rate = 0.0
        if self._universe_mask is not None:
            member_days = int(self._universe_mask.sum())
            missing_days = int((self._universe_mask & ~present).sum())
            real_missing = int((self._universe_mask & ~self._present).sum())
            rate = missing_days / member_days if member_days else 0.0
            real_rate = real_missing / member_days if member_days else 0.0

        return PanelDiagnostics(
            n_sessions=self._n_sessions,
            n_symbols=self._n_symbols,
            n_rows=n_rows,
            universe_member_days=member_days,
            missing_member_days=missing_days,
            missing_member_rate=rate,
            real_missing_member_rate=real_rate,
            carried_forward_days=carried,
            max_abs_session_return=largest,
        )
