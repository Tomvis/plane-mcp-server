"""Plane Community Edition cannot filter, count, or list work items workspace-wide.

CE v1.4.x 400s any `pql` ("unsupported_parameters") and 404s the workspace
list, the count endpoint and the workspace state catalogue. The tools fall back
to fetching project rows and evaluating PQL themselves (`pql_eval`).
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from plane.errors.errors import HttpError

from plane_mcp.pql_eval import Context, PqlError, compile_pql
from plane_mcp.tools import state as state_module
from plane_mcp.tools import workitem as workitem_module

TODAY = date(2026, 10, 13)
ME = "user-me"
GROUPS = {"s-backlog": "backlog", "s-started": "started", "s-done": "completed"}


def row(n, project="p-hw", **fields):
    return {
        "id": f"w{n}",
        "sequence_id": n,
        "project": project,
        "name": f"item {n}",
        "priority": "none",
        "state": "s-backlog",
        "assignees": [],
        "labels": [],
        "parent": None,
        "target_date": None,
        "start_date": None,
        "created_at": f"2026-10-{n:02d}T10:00:00Z",
        "updated_at": f"2026-10-{n:02d}T10:00:00Z",
        "is_draft": False,
        "archived_at": None,
        **fields,
    }


ROWS = [
    row(1, priority="high", state="s-started", assignees=[ME], target_date="2026-10-01"),
    row(2, priority="urgent", state="s-done", labels=["l-bug"], parent="w1", description_stripped="needle here"),
    row(3, priority="low", target_date="2026-10-20", start_date="2026-10-02", parent="w1"),
    row(4, project="p-net", priority="high", assignees=["someone"], name="Router capex"),
]


def ctx(rows=ROWS):
    return Context(
        state_group=GROUPS.get,
        current_user=lambda: ME,
        identifiers={"p-hw": "HW", "p-net": "NET"},
        rows=rows,
        today=TODAY,
    )


def ids(pql, rows=ROWS):
    predicate = compile_pql(pql, ctx(rows))
    return [r["id"] for r in rows if predicate(r)]


@pytest.mark.parametrize(
    ("pql", "expected"),
    [
        ('priority = "high"', ["w1", "w4"]),
        ('priority IN ("high","urgent")', ["w1", "w2", "w4"]),
        ('priority NOT IN ("high","urgent")', ["w3"]),
        ("stateGroup IN openStates()", ["w1", "w3", "w4"]),
        ('stateGroup = "COMPLETED"', ["w2"]),
        ("assignee = currentUser()", ["w1"]),
        ("hasNoAssignee()", ["w2", "w3"]),
        ('label = "l-bug"', ["w2"]),
        ("label IS EMPTY", ["w1", "w3", "w4"]),
        ("isOverdue()", ["w1"]),
        ("dueDate < today()", ["w1"]),
        ("dueDate IS NULL", ["w2", "w4"]),
        ('dueDate BETWEEN ("2026-10-15", daysFromNow(30))', ["w3"]),
        ('dueDate BETWEEN "2026-09-01" AND "2026-10-05"', ["w1"]),
        ("createdAt >= daysAgo(10)", ["w3", "w4"]),
        ('updatedAt < "2026-10-03"', ["w1", "w2"]),
        ('title ~ "CAPEX"', ["w4"]),
        ('text ~ "needle"', ["w2"]),
        ('id = "hw-2"', ["w2"]),
        ('id ~ "NET"', ["w4"]),
        ('project = "p-net"', ["w4"]),
        ('childOf("HW-1")', ["w2", "w3"]),
        ('parentOf("w3")', ["w1"]),
        ("hasChildren()", ["w1"]),
        ("isSubWorkItem() AND hasStartAndDueDates()", ["w3"]),
        ('priority = "high" OR priority = "low" AND stateGroup = "backlog"', ["w1", "w3", "w4"]),
        ('(priority = "high" OR priority = "low") AND NOT project = "p-net"', ["w1", "w3"]),
        ("isDraft = false AND NOT isArchived()", ["w1", "w2", "w3", "w4"]),
    ],
)
def test_evaluates(pql, expected):
    assert ids(pql) == expected


@pytest.mark.parametrize(
    ("pql", "reason"),
    [
        ('cycle = "c1"', "cycle list_work_items"),
        ("cycle IN activeCycle()", "cycle list_work_items"),
        ('cf["x"] = 1', "custom properties"),
        ('blocks("HW-1")', "workitem_relation"),
        ("isEpic()", "not supported"),
        ('colour = "red"', "unknown field"),
        ('priority = "high" AND', "expected"),
        ('childOf("HW-99")', "not among"),
        ('label > "x"', "does not apply"),
        ("daysAgo(x)", "expected"),
    ],
)
def test_refuses_with_a_reason(pql, reason):
    with pytest.raises(PqlError, match=reason):
        ids(pql)


# ---------------------------------------------------------------------------
# Dispatch against a CE-shaped client
# ---------------------------------------------------------------------------

UNSUPPORTED = HttpError(
    "Bad Request",
    status_code=400,
    response={"pql": "PQL and structured filters are not supported", "unsupported_parameters": ["pql"]},
)
NOT_FOUND = HttpError("Not Found", status_code=404, response={"error": "Page not found."})


def _page(items, cursor, size):
    start = int(cursor.split(":")[1]) * size if cursor else 0
    chunk = items[start : start + size]
    return SimpleNamespace(
        results=chunk, next_page_results=start + size < len(items), next_cursor=f"{size}:{start // size + 1}:0"
    )


class _WorkItems:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def list(self, workspace_slug, project_id, params=None):
        if params and params.pql:
            raise UNSUPPORTED
        self.calls += 1
        # A small server page size proves the fallback follows cursors.
        return _page([r for r in self.rows if r["project"] == project_id], params.cursor, 2)

    def list_workspace(self, workspace_slug, params=None):
        raise NOT_FOUND

    def count_workspace(self, workspace_slug, params=None):
        raise NOT_FOUND


class _CE:
    def __init__(self, rows=ROWS):
        self.work_items = _WorkItems(rows)
        self.projects = SimpleNamespace(
            list=lambda workspace_slug, params=None: _page(
                [{"id": "p-hw", "identifier": "HW"}, {"id": "p-net", "identifier": "NET"}], params.cursor, 100
            )
        )
        states = [{"id": k, "group": v, "name": v} for k, v in GROUPS.items()]
        self.states = SimpleNamespace(list=lambda workspace_slug, project_id, params=None: _page(states, None, 100))
        self.workspace_states = SimpleNamespace(
            list=lambda **kwargs: (_ for _ in ()).throw(NOT_FOUND),
        )
        self.users = SimpleNamespace(get_me=lambda: SimpleNamespace(id=ME))


@pytest.fixture
def ce(monkeypatch):
    client = _CE()
    for module in (workitem_module, state_module):
        monkeypatch.setattr(module, "get_plane_client_context", lambda: (client, "home"))
    return client


def _workitem(registered, **kwargs):
    return registered["workitem"].fn(**kwargs)


def test_project_list_with_pql_filters_client_side(registered, ce):
    result = _workitem(registered, action="list", project_id="p-hw", pql='priority IN ("high", "urgent")')

    assert [r["id"] for r in result["results"]] == ["w2", "w1"]  # -created_at, Plane's default
    assert result["total_count"] == 2
    assert "Community Edition" in result["note"]


def test_workspace_list_fans_out_over_projects(registered, ce):
    result = _workitem(registered, action="list", pql="stateGroup IN openStates()", order_by="sequence_id")

    assert [r["id"] for r in result["results"]] == ["w1", "w3", "w4"]


def test_workspace_list_without_pql_pages_and_honours_fields(registered, ce):
    first = _workitem(registered, action="list", per_page=3, fields="id,name")
    second = _workitem(registered, action="list", cursor=first["next_cursor"])

    assert first["results"][0] == {"id": "w4", "name": "Router capex"}
    assert (first["count"], first["next_page_results"], first["total_count"]) == (3, True, 4)
    assert [r["id"] for r in second["results"]] == ["w1"]
    assert second["next_page_results"] is False


def test_bad_pql_returns_a_correctable_error(registered, ce):
    result = _workitem(registered, action="list", project_id="p-hw", pql='cycle = "c1"')

    assert result["failed_pql"] == 'cycle = "c1"'
    assert "cycle list_work_items" in result["error"]
    assert "pql_reference" in result


def test_count_groups_client_side(registered, ce):
    result = _workitem(registered, action="count", pql='priority != "low"', group_by="state__group")

    assert result["total_count"] == 3
    assert result["grouped_counts"] == {"started": {"count": 1}, "completed": {"count": 1}, "backlog": {"count": 1}}


def test_count_with_project_and_sub_group(registered, ce):
    result = _workitem(registered, action="count", project_id="p-hw", group_by="priority", sub_group_by="labels__id")

    assert result["total_count"] == 3
    assert result["grouped_counts"]["urgent"] == {"count": 1, "sub_grouped_counts": {"l-bug": {"count": 1}}}
    assert result["grouped_counts"]["high"]["sub_grouped_counts"] == {"None": {"count": 1}}


def test_count_refuses_a_dimension_ce_cannot_compute(registered, ce):
    result = _workitem(registered, action="count", group_by="cycle_id")

    assert isinstance(result, str) and "cycle_id" in result and ce.work_items.calls == 0


def test_workspace_state_list_fans_out(registered, ce):
    result = registered["state"].fn(action="list")

    assert result["total_count"] == 6  # 3 states x 2 projects
    assert "every project's states" in result["note"]
