"""Map high-level repeat options onto OmniFocus repetition fields.

OmniFocus 4 serializes task recurrence with three on-disk fields. The exact tokens below
were captured from real OmniFocus 4 clients writing to a live sync bundle (the AppleScript
``sdef`` names differ from the serialized tokens, so they are NOT used here):

- ``repetition-rule``: an iCalendar RRULE string, e.g. ``FREQ=DAILY;INTERVAL=30``.
- ``repetition-schedule-type``: ``from-assigned`` (next occurrence relative to the assigned
  dates) or ``from-completion`` (next occurrence computed when the item is completed/dropped).
- ``repetition-anchor-date``: which date drives the next occurrence, as a token —
  ``dateDue`` / ``dateToStart`` / ``datePlanned``.
- ``repetition-method``: the legacy field (``fixed`` / ``due-after-completion`` /
  ``start-after-completion``); emitted alongside as a lossy mirror.

This module exposes a small vocabulary (``repeat_every`` / ``repeat_from``) and also accepts a raw
``repetition_rule`` / ``repetition_method`` for callers that already know the exact OmniFocus
tokens. Clearing an existing repetition is intentionally out of scope.
"""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

from dataclasses import dataclass

from omnifocus.errors import OFHTTPError

_UNIT_TO_FREQ = {"d": "DAILY", "w": "WEEKLY", "m": "MONTHLY", "y": "YEARLY"}

# ``repeat_from`` vocabulary -> (schedule-type, anchor-date token, legacy method mirror).
_REPEAT_FROM: dict[str, tuple[str, str, str]] = {
    "fixed": ("from-assigned", "dateDue", "fixed"),
    "due": ("from-assigned", "dateDue", "fixed"),
    "defer": ("from-assigned", "dateToStart", "fixed"),
    "completion": ("from-completion", "dateDue", "due-after-completion"),
    "due-after-completion": ("from-completion", "dateDue", "due-after-completion"),
    "start-after-completion": ("from-completion", "dateToStart", "start-after-completion"),
}


@dataclass(frozen=True)
class RepetitionFields:
    """Resolved OmniFocus repetition fields for a task."""

    repetition_rule: str
    repetition_method: str | None
    repetition_schedule_type: str | None
    repetition_anchor_date: str | None


def _rrule_from_repeat_every(repeat_every: str) -> str:
    """Translate a token like ``30d`` into an iCalendar RRULE string."""
    token = repeat_every.strip().lower()
    unit = token[-1:] if token else ""
    count = token[:-1]
    if unit not in _UNIT_TO_FREQ or not count.isdigit() or int(count) < 1:
        raise OFHTTPError(
            f"Invalid repeat_every: {repeat_every!r} (expected e.g. '30d', '6w', '3m', '1y')",
            status_code=422,
            code="validation_error",
        )
    return f"FREQ={_UNIT_TO_FREQ[unit]};INTERVAL={int(count)}"


def build_repetition(
    *,
    repeat_every: str | None = None,
    repeat_from: str | None = None,
    repetition_rule: str | None = None,
    repetition_method: str | None = None,
) -> RepetitionFields | None:
    """Resolve repetition inputs into concrete OmniFocus fields.

    Returns ``None`` when no recurrence was requested. A raw ``repetition_rule`` takes precedence
    over ``repeat_every``. ``repeat_from`` selects the schedule type and anchor-date token; an
    explicit ``repetition_method`` overrides the derived legacy-method mirror.
    """
    if repetition_rule:
        rule = repetition_rule
    elif repeat_every:
        rule = _rrule_from_repeat_every(repeat_every)
    else:
        if repeat_from is not None or repetition_method:
            raise OFHTTPError(
                "repeat_from/repetition_method require repeat_every or repetition_rule",
                status_code=422,
                code="validation_error",
            )
        return None

    schedule_key = repeat_from if repeat_from is not None else "fixed"
    if schedule_key not in _REPEAT_FROM:
        allowed = ", ".join(sorted(_REPEAT_FROM))
        raise OFHTTPError(
            f"Invalid repeat_from: {repeat_from!r} (expected one of: {allowed})",
            status_code=422,
            code="validation_error",
        )
    schedule_type, anchor_date, derived_method = _REPEAT_FROM[schedule_key]
    method = repetition_method if repetition_method else derived_method
    return RepetitionFields(
        repetition_rule=rule,
        repetition_method=method,
        repetition_schedule_type=schedule_type,
        repetition_anchor_date=anchor_date,
    )
