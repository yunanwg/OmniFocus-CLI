"""Tests for the high-level recurrence mapping."""

from __future__ import annotations

import pytest

from omnifocus.errors import OFHTTPError
from omnifocus.recurrence import RepetitionFields, build_repetition


class TestBuildRepetition:
    def test_returns_none_when_nothing_requested(self) -> None:
        assert build_repetition() is None

    @pytest.mark.parametrize(
        ("token", "expected_rule"),
        [
            ("30d", "FREQ=DAILY;INTERVAL=30"),
            ("6w", "FREQ=WEEKLY;INTERVAL=6"),
            ("3m", "FREQ=MONTHLY;INTERVAL=3"),
            ("1y", "FREQ=YEARLY;INTERVAL=1"),
            ("45D", "FREQ=DAILY;INTERVAL=45"),
        ],
    )
    def test_repeat_every_maps_to_rrule(self, token: str, expected_rule: str) -> None:
        # A plain calendar repeat (no repeat_from) defaults to the "from-assigned"
        # schedule anchored on the due date, with the legacy "fixed" method mirror.
        # Tokens captured from a real OmniFocus 4 client writing to a live bundle.
        result = build_repetition(repeat_every=token)
        assert result == RepetitionFields(
            repetition_rule=expected_rule,
            repetition_method="fixed",
            repetition_schedule_type="from-assigned",
            repetition_anchor_date="dateDue",
        )

    def test_completion_sets_method_and_schedule_type(self) -> None:
        result = build_repetition(repeat_every="30d", repeat_from="completion")
        assert result == RepetitionFields(
            repetition_rule="FREQ=DAILY;INTERVAL=30",
            repetition_method="due-after-completion",
            repetition_schedule_type="from-completion",
            repetition_anchor_date="dateDue",
        )

    def test_start_after_completion(self) -> None:
        result = build_repetition(repeat_every="1w", repeat_from="start-after-completion")
        assert result == RepetitionFields(
            repetition_rule="FREQ=WEEKLY;INTERVAL=1",
            repetition_method="start-after-completion",
            repetition_schedule_type="from-completion",
            repetition_anchor_date="dateToStart",
        )

    def test_defer_anchors_on_defer_date(self) -> None:
        result = build_repetition(repeat_every="2w", repeat_from="defer")
        assert result == RepetitionFields(
            repetition_rule="FREQ=WEEKLY;INTERVAL=2",
            repetition_method="fixed",
            repetition_schedule_type="from-assigned",
            repetition_anchor_date="dateToStart",
        )

    def test_raw_rule_takes_precedence_over_repeat_every(self) -> None:
        result = build_repetition(repeat_every="30d", repetition_rule="FREQ=HOURLY;INTERVAL=2")
        assert result is not None
        assert result.repetition_rule == "FREQ=HOURLY;INTERVAL=2"

    def test_explicit_method_override(self) -> None:
        result = build_repetition(
            repetition_rule="FREQ=DAILY", repetition_method="start-after-completion"
        )
        assert result is not None
        assert result.repetition_method == "start-after-completion"

    @pytest.mark.parametrize("token", ["30", "d", "0d", "5x", "-3d"])
    def test_invalid_repeat_every_raises(self, token: str) -> None:
        with pytest.raises(OFHTTPError, match="Invalid repeat_every"):
            build_repetition(repeat_every=token)

    def test_empty_repeat_every_is_noop(self) -> None:
        assert build_repetition(repeat_every="") is None

    def test_invalid_repeat_from_raises(self) -> None:
        with pytest.raises(OFHTTPError, match="Invalid repeat_from"):
            build_repetition(repeat_every="30d", repeat_from="whenever")

    def test_repeat_from_without_rule_raises(self) -> None:
        with pytest.raises(OFHTTPError, match="require repeat_every or repetition_rule"):
            build_repetition(repeat_from="completion")

    def test_method_without_rule_raises(self) -> None:
        with pytest.raises(OFHTTPError, match="require repeat_every or repetition_rule"):
            build_repetition(repetition_method="fixed")
