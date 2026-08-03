"""Participant-flow parsing and flow-state labelling.

The parsing tests all describe one failure mode: NSE returning something that is not the file you
asked for, in a shape naive code accepts. The labelling tests describe a different one: a state
that knows its own future.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.capacity.flows import (
    FlowStateRule,
    label_flow_state,
    net_position_series,
    parse_participant_csv,
)
from src.common.exceptions import DataIntegrityError

#: A faithful reduction of the real file: doubled-quoted title line, one row per category, a TOTAL
#: row, and the two total columns the net series is built from.
REAL_SHAPE = (
    '""Participant wise Trading Volume (no. of contracts) in Equity Derivatives as on '
    'Jun 15,2021"",,,\n'
    "Client Type,Future Index Long,Future Index Short,Total Long Contracts,Total Short Contracts\n"
    "Client,98150,95438,12909807,12901422\n"
    "DII,20,0,36577,35985\n"
    "FII,25298,25278,3626889,3624361\n"
    "Pro,53881,56633,14931682,14943187\n"
    "TOTAL,177349,177349,31504955,31504955\n"
)
SESSION = date(2021, 6, 15)


def test_a_real_shaped_file_parses_into_one_row_per_category_and_measure() -> None:
    frame = parse_participant_csv(REAL_SHAPE, SESSION)
    assert frame["category"].n_unique() == 4
    assert frame["measure"].n_unique() == 4
    assert frame.height == 16
    fii_long = frame.filter(
        (pl.col("category") == "FII") & (pl.col("measure") == "Total Long Contracts")
    )
    assert fii_long["contracts"][0] == pytest.approx(3_626_889)


def test_the_total_row_is_dropped_so_nothing_downstream_double_counts() -> None:
    frame = parse_participant_csv(REAL_SHAPE, SESSION)
    assert "TOTAL" not in frame["category"].to_list()


def test_an_html_error_page_is_refused_rather_than_parsed_into_an_empty_frame() -> None:
    with pytest.raises(DataIntegrityError, match="HTML"):
        parse_participant_csv("<!DOCTYPE html>\n<html><body>404</body></html>", SESSION)


def test_a_file_missing_a_participant_category_raises() -> None:
    """A layout change that dropped DII would otherwise pass every structural check."""
    without_dii = "\n".join(
        line for line in REAL_SHAPE.splitlines() if not line.startswith("DII")
    )
    with pytest.raises(DataIntegrityError, match="missing categories"):
        parse_participant_csv(without_dii + "\n", SESSION)


def test_an_empty_cell_is_not_read_as_zero() -> None:
    """"NSE did not report this" and "no contracts traded" are different facts."""
    blanked = REAL_SHAPE.replace("FII,25298,25278,3626889,3624361",
                                 "FII,,25278,3626889,3624361")
    with pytest.raises(DataIntegrityError, match="empty cell"):
        parse_participant_csv(blanked, SESSION)


def test_net_position_is_long_minus_short_from_the_total_columns() -> None:
    frame = parse_participant_csv(REAL_SHAPE, SESSION)
    net = net_position_series(frame, category="FII")
    assert net.height == 1
    assert net["net"][0] == pytest.approx(3_626_889 - 3_624_361)


def test_net_position_for_an_absent_category_raises_instead_of_returning_empty() -> None:
    frame = parse_participant_csv(REAL_SHAPE, SESSION)
    with pytest.raises(DataIntegrityError):
        net_position_series(frame, category="Retail")


# --- flow-state labelling -------------------------------------------------------------------


def _net_series(values: list[float]) -> pl.DataFrame:
    start = date(2020, 1, 1)
    return pl.DataFrame(
        {
            "session_date": [start + timedelta(days=i) for i in range(len(values))],
            "net": values,
        }
    )


def test_a_flow_state_never_uses_the_future_of_its_own_series() -> None:
    """Truncating the series must not change any label that survives the truncation.

    This is the leakage test. A baseline computed over the whole series would make every early
    label depend on flows that had not happened, and the symptom — a capacity comparison
    conditioned on information the trader did not have — would be invisible in the output.
    """
    rng = np.random.default_rng(11)
    values = list(rng.normal(0, 1000, size=400))
    rule = FlowStateRule(window=5, baseline=60, threshold=1.0)
    full = label_flow_state(_net_series(values), rule)
    truncated = label_flow_state(_net_series(values[:300]), rule)
    shared = full.head(300)["flow_state"].to_list()
    assert shared == truncated["flow_state"].to_list()


def test_the_burn_in_is_unlabelled_rather_than_labelled_from_a_short_window() -> None:
    rule = FlowStateRule(window=5, baseline=60, threshold=1.0)
    labelled = label_flow_state(_net_series([100.0] * 200), rule)
    assert labelled["flow_state"][:60].null_count() == 60


def test_a_sustained_surge_is_labelled_inflow_and_its_mirror_outflow() -> None:
    rng = np.random.default_rng(5)
    quiet = list(rng.normal(0, 100, size=300))
    rule = FlowStateRule(window=5, baseline=60, threshold=1.0)

    surge = label_flow_state(_net_series(quiet + [5000.0] * 10), rule)
    assert surge["flow_state"].to_list()[-1] == "inflow"

    drain = label_flow_state(_net_series(quiet + [-5000.0] * 10), rule)
    assert drain["flow_state"].to_list()[-1] == "outflow"


def test_a_flat_series_is_left_unlabelled_rather_than_divided_by_its_zero_spread() -> None:
    """A series with no variation has no scale to standardise against, so it gets no state.

    Refusing to label is the right answer and not an obvious one — the tempting alternative is to
    call it neutral, which would assert that flows were unremarkable when what actually happened
    is that the measurement was undefined. The two are different claims and only one is true.
    """
    rule = FlowStateRule(window=5, baseline=60, threshold=1.0)
    labelled = label_flow_state(_net_series([100.0] * 200), rule)
    assert labelled["flow_state"].null_count() == labelled.height


def test_a_frame_without_the_required_columns_raises() -> None:
    with pytest.raises(DataIntegrityError):
        label_flow_state(pl.DataFrame({"session_date": [date(2020, 1, 1)]}), FlowStateRule())
