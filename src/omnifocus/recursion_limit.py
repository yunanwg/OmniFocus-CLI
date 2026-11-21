"""Shared recursion-limit guard for deep OmniFocus traversals.

Real OmniFocus bundles have deep delta DAG chains (one delta per sync since the
last compaction) and deeply nested ``ElementTree`` documents. Both the sync
graph walks (:mod:`omnifocus.sync.graph`) and the bundle parser
(:mod:`omnifocus.parser`) recurse over these structures, and the interpreter's
default limit of 1000 is not enough.
"""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

import sys

MIN_RECURSION_LIMIT = 50_000


def ensure_recursion_limit() -> None:
    """Raise the interpreter recursion limit to the needed floor if it is lower.

    Never lowers a limit already configured higher by the host process.
    """
    if sys.getrecursionlimit() < MIN_RECURSION_LIMIT:
        sys.setrecursionlimit(MIN_RECURSION_LIMIT)
