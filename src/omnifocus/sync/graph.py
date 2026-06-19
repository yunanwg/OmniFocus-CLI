"""OmniFocus sync DAG traversal helpers.

This module contains the pure graph logic used to reconstruct the current
read-view of an OmniFocus bundle. The WebDAV bundle is a DAG of delta ZIPs:
each delta produces a new tail and may merge multiple parent tails.
"""

from __future__ import annotations

__author__ = "Maciej Szymczak <maciej@szymczak.at>"

from omnifocus.errors import OFError
from omnifocus.sync.client_state import ClientStateDocument
from omnifocus.sync.protocol import BundleState


def current_tail_id(
    state: BundleState,
    remote_clients: dict[str, ClientStateDocument],
) -> str | None:
    """Return the best-known current tail identifier for the bundle."""
    frontier = current_frontier_tail_ids(state, remote_clients)
    return frontier[0] if frontier else None


def delta_derived_frontier_tail_ids(state: BundleState) -> tuple[str, ...]:
    """Return the frontier tails implied purely by the delta DAG.

    These are the maximal sink tails (delta tails no other delta lists as a parent)
    reachable from the baseline — the frontier the read model is built from in the
    normal case, independent of client state. Empty when no such tail exists, in
    which case the frontier must fall back to client-advertised tails (see
    :func:`current_frontier_tail_ids`).
    """
    if not (state.deltas and state.baseline.tail_id is not None):
        return ()
    sink_tails = tuple(
        delta.tail_id
        for delta in reversed(state.deltas)
        if is_frontier_tail(state, delta.tail_id)
        and tail_reachable_from_baseline(state, delta.tail_id)
    )
    return maximal_tail_ids(state, sink_tails)


def current_frontier_tail_ids(
    state: BundleState,
    remote_clients: dict[str, ClientStateDocument],
) -> tuple[str, ...]:
    """Return the current frontier tails used to build the read model."""
    delta_frontier = delta_derived_frontier_tail_ids(state)
    if delta_frontier:
        return delta_frontier

    reachable_client_tails: list[str] = []
    for ref in reversed(state.clients):
        document = remote_clients.get(ref.client_id)
        if document is None:
            continue
        for candidate in document.tail_identifiers:
            if candidate not in reachable_client_tails and tail_reachable_from_baseline(
                state, candidate
            ):
                reachable_client_tails.append(candidate)

    maximal_client_tails = maximal_tail_ids(state, tuple(reachable_client_tails))
    if maximal_client_tails:
        return maximal_client_tails

    if state.deltas and state.baseline.tail_id is not None:
        return (state.baseline.tail_id,)
    if state.deltas:
        return (state.deltas[-1].tail_id,)
    if state.baseline.tail_id:
        return (state.baseline.tail_id,)
    return ()


def transaction_filenames_for_frontier(
    state: BundleState,
    frontier_tail_identifiers: tuple[str, ...],
) -> list[str]:
    """Return delta filenames reachable from the given frontier tails."""
    if not frontier_tail_identifiers:
        return []
    selected_tail_ids = reachable_delta_tail_ids(state, frontier_tail_identifiers)
    return topologically_sorted_delta_filenames(state, selected_tail_ids)


def is_frontier_tail(state: BundleState, tail_id: str) -> bool:
    """Return whether *tail_id* is not referenced as a parent by any delta."""
    all_parent_tail_ids = {
        parent_tail_id for delta in state.deltas for parent_tail_id in delta.parent_tail_ids
    }
    return tail_id not in all_parent_tail_ids


def topologically_sorted_delta_filenames(
    state: BundleState,
    selected_tail_ids: set[str],
) -> list[str]:
    """Return selected delta filenames in parent-before-child order."""
    deltas_by_tail = {delta.tail_id: delta for delta in state.deltas}
    ordered: list[str] = []
    visited: set[str] = set()

    # Iterative post-order DFS with an explicit stack (not recursion) so a long
    # delta chain cannot exhaust the Python call stack. Each tail is pushed once
    # to expand its parents (``emit=False``) and once to be appended after every
    # parent has been emitted (``emit=True``).
    for root in state.deltas:
        if root.tail_id not in selected_tail_ids:
            continue
        stack: list[tuple[str, bool]] = [(root.tail_id, False)]
        while stack:
            tail_id, emit = stack.pop()
            if tail_id not in selected_tail_ids:
                continue
            if emit:
                visited.add(tail_id)
                ordered.append(deltas_by_tail[tail_id].filename)
                continue
            if tail_id in visited:
                continue
            stack.append((tail_id, True))
            for parent_tail_id in reversed(deltas_by_tail[tail_id].parent_tail_ids):
                stack.append((parent_tail_id, False))
    return ordered


def maximal_tail_ids(
    state: BundleState,
    tail_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Return tails with ancestor tails removed while preserving order."""
    unique_tail_ids = tuple(dict.fromkeys(tail_ids))
    maximal: list[str] = []
    for candidate in unique_tail_ids:
        if any(
            other != candidate and tail_depends_on(state, other, candidate)
            for other in unique_tail_ids
        ):
            continue
        maximal.append(candidate)
    return tuple(maximal)


def reachable_delta_tail_ids(
    state: BundleState,
    frontier_tail_identifiers: tuple[str, ...],
) -> set[str]:
    """Return the closure of delta tails reachable from the frontier."""
    baseline_tail = state.baseline.tail_id
    deltas_by_tail = {delta.tail_id: delta for delta in state.deltas}
    selected: set[str] = set()
    on_path: set[str] = set()

    # Iterative DFS with an explicit stack (not recursion) so a long chain cannot
    # exhaust the Python call stack. ``on_path`` holds the tails on the current
    # root-to-node path for cycle detection: a tail is added when first expanded
    # and removed once all its parents are selected (the ``done=True`` marker).
    for start in frontier_tail_identifiers:
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            tail_id, done = stack.pop()
            if done:
                on_path.discard(tail_id)
                selected.add(tail_id)
                continue
            if tail_id == baseline_tail or tail_id in selected:
                continue
            if tail_id in on_path:
                raise OFError(f"Detected a cycle while resolving delta DAG at tail {tail_id!r}")
            delta = deltas_by_tail.get(tail_id)
            if delta is None:
                raise OFError(
                    f"Could not resolve the current OmniFocus delta DAG for tail {tail_id!r}"
                )
            on_path.add(tail_id)
            stack.append((tail_id, True))
            for parent_tail_id in reversed(delta.parent_tail_ids):
                stack.append((parent_tail_id, False))

    return selected


def tail_reachable_from_baseline(state: BundleState, tail_id: str) -> bool:
    """Return whether *tail_id* can be walked back to the baseline tail."""
    baseline_tail = state.baseline.tail_id
    if baseline_tail is None or not state.deltas:
        return True
    if tail_id == baseline_tail:
        return True

    deltas_by_tail = {delta.tail_id: delta for delta in state.deltas}
    memo: dict[str, bool] = {}
    on_path: set[str] = set()

    # Iterative post-order DFS with an explicit stack (not recursion) so a long
    # chain cannot exhaust the Python call stack. A tail is reachable iff it is
    # the baseline or every parent is reachable; a tail re-entered while still on
    # the current path (a cycle) or whose delta is missing is unreachable.
    stack: list[tuple[str, bool]] = [(tail_id, False)]
    while stack:
        cursor, resolved = stack.pop()
        if resolved:
            memo[cursor] = all(memo[parent] for parent in deltas_by_tail[cursor].parent_tail_ids)
            on_path.discard(cursor)
            continue
        if cursor == baseline_tail:
            memo[cursor] = True
            continue
        if cursor in memo:
            continue
        if cursor in on_path:
            memo[cursor] = False
            continue
        delta = deltas_by_tail.get(cursor)
        if delta is None:
            memo[cursor] = False
            continue
        on_path.add(cursor)
        stack.append((cursor, True))
        for parent in delta.parent_tail_ids:
            stack.append((parent, False))

    return memo[tail_id]


def tail_depends_on(state: BundleState, tail_id: str, ancestor_tail_id: str) -> bool:
    """Return whether *tail_id* transitively depends on *ancestor_tail_id*."""
    if tail_id == ancestor_tail_id:
        return False
    deltas_by_tail = {delta.tail_id: delta for delta in state.deltas}
    visited: set[str] = set()

    # Iterative DFS with an explicit stack (not recursion) so a long chain cannot
    # exhaust the Python call stack; returns as soon as a delta listing
    # ``ancestor_tail_id`` among its parents is reached.
    stack: list[str] = [tail_id]
    while stack:
        cursor = stack.pop()
        if cursor in visited:
            continue
        visited.add(cursor)
        delta = deltas_by_tail.get(cursor)
        if delta is None:
            continue
        if ancestor_tail_id in delta.parent_tail_ids:
            return True
        stack.extend(delta.parent_tail_ids)

    return False
