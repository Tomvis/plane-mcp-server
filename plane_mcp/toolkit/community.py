"""Plane Community Edition fallbacks for listing and counting work items.

CE v1.4.x answers any `pql`/`filters` parameter with a 400 naming
`unsupported_parameters`, and has no workspace-scope work item list, work item
count (at any scope) or workspace state list -- those 404. Each caller tries
the real endpoint first; on one of those refusals it lands here, which fetches
the project rows, filters them with `pql_eval`, then orders, counts and pages
them the way Plane would. Every answer carries `note` so the caller knows the
result was computed by this server, not by Plane.

Nothing here fires against an edition that serves the endpoints.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from fastmcp.utilities.logging import get_logger
from plane.errors.errors import HttpError
from plane.models.query_params import PaginatedQueryParams, WorkItemQueryParams

from plane_mcp.pql_eval import Context, PqlError, Row, compile_pql, ref_id, ref_ids
from plane_mcp.pql_reference import PQL_FULL_REFERENCE

logger = get_logger(__name__)

NOTE = "Plane Community Edition: computed by the MCP server from the project rows, since CE cannot filter or count."

FETCH_PAGE = 100
MAX_PAGES = 100  # 10k rows per project; a stuck cursor must not loop forever
DEFAULT_PAGE = 100

PRIORITY_RANK = {"urgent": 0, "high": 1, "medium": 2, "low": 3, "none": 4}


def rejects_filters(exc: HttpError) -> bool:
    """CE's 400 for a pql/filters parameter it will not evaluate."""
    body = exc.response
    return exc.status_code == 400 and isinstance(body, dict) and "unsupported_parameters" in body


def not_found(exc: HttpError) -> bool:
    return exc.status_code == 404


def _dump(item: Any) -> Any:
    return item.model_dump() if hasattr(item, "model_dump") else item


def _pages(fetch: Callable[[str | None], Any]) -> list[Any]:
    """Every row behind a cursor-paginated endpoint."""
    rows: list[Any] = []
    cursor = None
    for _ in range(MAX_PAGES):
        response = fetch(cursor)
        results = list(response.results or [])
        rows.extend(_dump(item) for item in results)
        if not results or not response.next_page_results:
            return rows
        cursor = response.next_cursor
    logger.warning("community fallback stopped after %d pages", MAX_PAGES)
    return rows


def fetch_work_items(method: Any, *, expand: str = "", **target: Any) -> list[Row]:
    """All work items one list endpoint returns, unfiltered."""
    return _pages(
        lambda cursor: method(
            **target, params=WorkItemQueryParams(per_page=FETCH_PAGE, cursor=cursor, expand=expand or None)
        )
    )


def projects(client: Any, workspace_slug: str) -> list[Row]:
    return _pages(
        lambda cursor: client.projects.list(
            workspace_slug=workspace_slug, params=PaginatedQueryParams(per_page=FETCH_PAGE, cursor=cursor)
        )
    )


def project_states(client: Any, workspace_slug: str, project_id: str) -> list[Row]:
    return _pages(
        lambda cursor: client.states.list(
            workspace_slug=workspace_slug,
            project_id=project_id,
            params={"per_page": FETCH_PAGE, **({"cursor": cursor} if cursor else {})},
        )
    )


def context(client: Any, workspace_slug: str, rows: list[Row], known_projects: list[Row] | None = None) -> Context:
    """Lookups for `pql_eval`, each fetched only if the query needs it."""
    groups: dict[str, str] = {}
    loaded: set[str] = set()
    me: list[str | None] = []

    def state_group(state_id: str) -> str | None:
        if state_id not in groups:
            for project_id in {ref_id(row.get("project")) for row in rows} - loaded - {None}:
                loaded.add(project_id)
                groups.update({s["id"]: s.get("group") for s in project_states(client, workspace_slug, project_id)})
        return groups.get(state_id)

    def current_user() -> str | None:
        if not me:
            me.append(client.users.get_me().id)
        return me[0]

    listing = known_projects if known_projects is not None else projects(client, workspace_slug)
    return Context(
        state_group=state_group,
        current_user=current_user,
        identifiers={p["id"]: p.get("identifier") for p in listing},
        rows=rows,
    )


def pql_error(tool: str, action: str, pql: str, exc: PqlError) -> dict[str, Any]:
    return {
        "error": str(exc),
        "failed_pql": pql,
        "pql_reference": PQL_FULL_REFERENCE,
        "hint": (
            f"This Plane edition cannot filter, so the MCP server evaluates PQL itself. Fix the PQL and retry "
            f"{tool} {action}."
        ),
    }


def _order(rows: list[Row], order_by: str) -> list[Row]:
    key = order_by or "-created_at"
    name = key.lstrip("-")

    def rank(row: Row) -> Any:
        value = row.get(name)
        if name == "priority":
            return PRIORITY_RANK.get(value or "none", 5)
        return ref_id(value) if isinstance(value, dict) else value

    present = [row for row in rows if row.get(name) is not None]
    absent = [row for row in rows if row.get(name) is None]
    try:
        present.sort(key=rank, reverse=key.startswith("-"))
    except TypeError:
        present.sort(key=lambda row: str(rank(row)), reverse=key.startswith("-"))
    return present + absent


def page(rows: list[Any], cursor: str = "", per_page: int = 0, fields: str = "") -> dict[str, Any]:
    """Slice rows into Plane's envelope; cursors use Plane's `size:page:is_prev` shape."""
    size, index = per_page or DEFAULT_PAGE, 0
    if cursor:
        try:
            size, index = (int(part) for part in cursor.split(":")[:2])
        except ValueError:
            return {"error": f"invalid cursor {cursor!r}; pass back a next_cursor from a previous page"}
    size = max(size, 1)
    start = index * size
    chunk = rows[start : start + size]
    requested = {name.strip() for name in fields.split(",")} - {""} if fields else None
    if requested:
        chunk = [{k: v for k, v in row.items() if k in requested} for row in chunk]
    total = len(rows)
    return {
        "results": chunk,
        "total_count": total,
        "count": len(chunk),
        "next_cursor": f"{size}:{index + 1}:0",
        "prev_cursor": f"{size}:{index - 1}:1",
        "next_page_results": start + size < total,
        "prev_page_results": index > 0,
        "total_pages": -(-total // size),
        "note": NOTE,
    }


def _filter(
    client: Any, workspace_slug: str, rows: list[Row], pql: str, known_projects: list[Row] | None = None
) -> list[Row]:
    """Rows matching pql. Raises PqlError."""
    if not pql:
        return rows
    predicate = compile_pql(pql, context(client, workspace_slug, rows, known_projects))
    return [row for row in rows if predicate(row)]


def _project_rows(
    client: Any, workspace_slug: str, project_id: str, *, archived: bool = False, expand: str = ""
) -> tuple[list[Row], list[Row]]:
    """Work items of one project, or of every project; plus the project listing."""
    listing = projects(client, workspace_slug)
    method = client.work_items.list_archived if archived else client.work_items.list
    targets = [project_id] if project_id else [p["id"] for p in listing if not p.get("archived_at")]
    rows: list[Row] = []
    for target in targets:
        rows.extend(fetch_work_items(method, expand=expand, workspace_slug=workspace_slug, project_id=target))
    return rows, listing


def list_work_items(
    client: Any,
    workspace_slug: str,
    *,
    tool: str,
    action: str,
    project_id: str = "",
    archived: bool = False,
    pql: str = "",
    order_by: str = "",
    cursor: str = "",
    per_page: int = 0,
    expand: str = "",
    fields: str = "",
) -> dict[str, Any]:
    rows, listing = _project_rows(client, workspace_slug, project_id, archived=archived, expand=expand)
    try:
        matched = _filter(client, workspace_slug, rows, pql, listing)
    except PqlError as exc:
        return pql_error(tool, action, pql, exc)
    return page(_order(matched, order_by), cursor, per_page, fields)


def filter_page(
    client: Any,
    workspace_slug: str,
    method: Any,
    *,
    tool: str,
    action: str,
    pql: str,
    order_by: str = "",
    cursor: str = "",
    per_page: int = 0,
    expand: str = "",
    fields: str = "",
    **target: Any,
) -> dict[str, Any]:
    """The work items of a parent record (cycle, module), filtered here."""
    rows = fetch_work_items(method, expand=expand, **target)
    try:
        matched = _filter(client, workspace_slug, rows, pql)
    except PqlError as exc:
        return pql_error(tool, action, pql, exc)
    return page(_order(matched, order_by), cursor, per_page, fields)


GROUPS: dict[str, Callable[[Row, Context], Iterable[Any]]] = {
    "state_id": lambda r, c: [ref_id(r.get("state"))],
    "state__group": lambda r, c: [c.state_group(ref_id(r.get("state")) or "")],
    "priority": lambda r, c: [r.get("priority")],
    "project_id": lambda r, c: [ref_id(r.get("project"))],
    "type_id": lambda r, c: [ref_id(r.get("type_id")) or ref_id(r.get("type"))],
    "labels__id": lambda r, c: ref_ids(r.get("labels")) or [None],
    "assignees__id": lambda r, c: ref_ids(r.get("assignees")) or [None],
    "created_by": lambda r, c: [ref_id(r.get("created_by"))],
    "target_date": lambda r, c: [r.get("target_date")],
    "start_date": lambda r, c: [r.get("start_date")],
}


def _key(value: Any) -> str:
    return "None" if value is None else str(value)


def count_work_items(
    client: Any,
    workspace_slug: str,
    *,
    tool: str,
    action: str,
    project_id: str = "",
    pql: str = "",
    group_by: str = "",
    sub_group_by: str = "",
) -> dict[str, Any] | str:
    for dimension in filter(None, (group_by, sub_group_by)):
        if dimension not in GROUPS:
            return (
                f"Error: group_by {dimension} needs Plane's server-side count, which Community Edition lacks. "
                f"Available here: {', '.join(GROUPS)}."
            )
    rows, listing = _project_rows(client, workspace_slug, project_id)
    try:
        matched = _filter(client, workspace_slug, rows, pql, listing)
    except PqlError as exc:
        return pql_error(tool, action, pql, exc)
    ctx = context(client, workspace_slug, rows, listing)
    grouped: dict[str, dict[str, Any]] = {}
    if group_by:
        counts: Counter[str] = Counter()
        subs: dict[str, Counter[str]] = defaultdict(Counter)
        for row in matched:
            for value in GROUPS[group_by](row, ctx):
                counts[_key(value)] += 1
                if sub_group_by:
                    for sub in GROUPS[sub_group_by](row, ctx):
                        subs[_key(value)][_key(sub)] += 1
        for key, n in counts.items():
            grouped[key] = {"count": n}
            if sub_group_by:
                grouped[key]["sub_grouped_counts"] = {s: {"count": m} for s, m in subs[key].items()}
    return {
        "grouped_by": group_by or None,
        "sub_grouped_by": sub_group_by or None,
        "total_count": len(matched),
        "grouped_counts": grouped,
        "note": NOTE,
    }


def workspace_states(client: Any, workspace_slug: str, cursor: str = "", per_page: int = 0) -> dict[str, Any]:
    """Every project's states, since CE has no workspace state catalogue."""
    rows: list[Row] = []
    for project in projects(client, workspace_slug):
        if not project.get("archived_at"):
            rows.extend(project_states(client, workspace_slug, project["id"]))
    result = page(rows, cursor, per_page)
    result["note"] = "Plane Community Edition has no workspace states; these are every project's states."
    return result
