"""Evaluate PQL against work items client-side, for Plane Community Edition.

CE rejects every `pql` parameter with a 400 ("PQL and structured filters are
not supported on this Plane edition"). Rather than leave the filter unusable,
the fallback fetches the rows and filters them here. Only fields a work item
row actually carries can be evaluated; anything needing server-side joins
(cycle, module, milestone, mention, subscriber, custom properties, most
relation functions) is refused with a reason instead of silently matching
nothing.

`compile_pql(text, ctx)` returns a predicate over a dumped work item dict.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

Row = dict[str, Any]
Predicate = Callable[[Row], bool]


class PqlError(ValueError):
    """The PQL is malformed, or asks for something CE rows cannot answer."""


OPEN_STATES = ("backlog", "unstarted", "started")
CLOSED_STATES = ("completed", "cancelled")
ACTIVE_STATES = ("unstarted", "started")

NEEDS_SERVER = {
    "cycle": "list the cycle's work items with `cycle list_work_items` instead",
    "module": "list the module's work items with `module list_work_items` instead",
    "milestone": "milestones are a Commercial feature",
    "mention": "mentions are not on the work item row",
    "subscriber": "subscribers are not on the work item row",
    "teamspaceProject": "teamspaces are a Commercial feature",
}


# ---------------------------------------------------------------------------
# Context: what a row alone cannot tell
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """Lookups the evaluator needs beyond the row itself. All lazy."""

    state_group: Callable[[str], str | None] = lambda _state_id: None
    current_user: Callable[[], str | None] = lambda: None
    identifiers: dict[str, str] = field(default_factory=dict)  # project id -> "HW"
    rows: list[Row] = field(default_factory=list)  # the whole fetched set, for relation lookups
    today: date = field(default_factory=date.today)

    def identifier(self, row: Row) -> str | None:
        prefix = self.identifiers.get(ref_id(row.get("project")) or "")
        sequence = row.get("sequence_id")
        return f"{prefix}-{sequence}" if prefix and sequence is not None else None

    def resolve(self, ref: str) -> str:
        """A work item UUID from "HW-5" or a UUID, within the fetched rows."""
        for row in self.rows:
            if row.get("id") == ref or (self.identifier(row) or "").upper() == ref.upper():
                return row["id"]
        raise PqlError(f"work item {ref!r} is not among the work items fetched for this query")

    def parents(self) -> set[str]:
        return {ref_id(row.get("parent")) for row in self.rows if row.get("parent")}


def ref_id(value: Any) -> str | None:
    """An id from a bare UUID or an expanded object."""
    if isinstance(value, dict):
        return value.get("id")
    return value


def ref_ids(values: Any) -> list[str]:
    return [i for i in (ref_id(v) for v in values or []) if i]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
    \s*(?:
      (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
    | (?P<number>-?\d+(?:\.\d+)?)
    | (?P<op>!=|>=|<=|=|>|<|~)
    | (?P<punct>[(),\[\]])
    | (?P<word>[A-Za-z_][A-Za-z0-9_]*)
    )""",
    re.VERBOSE,
)

KEYWORDS = {"AND", "OR", "NOT", "IN", "IS", "NULL", "EMPTY", "BETWEEN", "TRUE", "FALSE"}


@dataclass
class Token:
    kind: str  # string | number | op | punct | word | kw | end
    value: Any


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match or match.end() == pos:
            raise PqlError(f"unexpected character at position {pos}: {text[pos : pos + 15]!r}")
        pos = match.end()
        kind = match.lastgroup
        raw = match.group(kind)
        if kind == "string":
            tokens.append(Token("string", re.sub(r"\\(.)", r"\1", raw[1:-1])))
        elif kind == "number":
            tokens.append(Token("number", float(raw) if "." in raw else int(raw)))
        elif kind == "word" and raw.upper() in KEYWORDS:
            tokens.append(Token("kw", raw.upper()))
        else:
            tokens.append(Token(kind, raw))
    tokens.append(Token("end", None))
    return tokens


# ---------------------------------------------------------------------------
# Values: literals and functions
# ---------------------------------------------------------------------------


def _months(day: date, n: int) -> date:
    month = day.month - 1 + n
    year, month = day.year + month // 12, month % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _value_function(name: str, args: list[Any], ctx: Context) -> Any:  # noqa: PLR0911, PLR0912
    today = ctx.today
    n = args[0] if args else None

    def needs_int() -> int:
        if not isinstance(n, int):
            raise PqlError(f"{name}() takes one whole number, e.g. {name}(7)")
        return n

    simple: dict[str, Callable[[], Any]] = {
        "today": lambda: today,
        "now": lambda: datetime.now(timezone.utc),
        "startOfDay": lambda: today,
        "endOfDay": lambda: today,
        "startOfWeek": lambda: today - timedelta(days=today.weekday()),
        "endOfWeek": lambda: today + timedelta(days=6 - today.weekday()),
        "startOfMonth": lambda: today.replace(day=1),
        "endOfMonth": lambda: today.replace(day=calendar.monthrange(today.year, today.month)[1]),
        "startOfYear": lambda: date(today.year, 1, 1),
        "endOfYear": lambda: date(today.year, 12, 31),
        "currentUser": ctx.current_user,
        "openStates": lambda: list(OPEN_STATES),
        "closedStates": lambda: list(CLOSED_STATES),
        "activeStates": lambda: list(ACTIVE_STATES),
    }
    if name in simple:
        if args:
            raise PqlError(f"{name}() takes no arguments")
        value = simple[name]()
        if name == "currentUser" and not value:
            raise PqlError("currentUser() could not be resolved for this API key")
        return value
    offsets: dict[str, Callable[[int], date]] = {
        "daysAgo": lambda k: today - timedelta(days=k),
        "daysFromNow": lambda k: today + timedelta(days=k),
        "weeksAgo": lambda k: today - timedelta(weeks=k),
        "weeksFromNow": lambda k: today + timedelta(weeks=k),
        "monthsAgo": lambda k: _months(today, -k),
        "monthsFromNow": lambda k: _months(today, k),
    }
    if name in offsets:
        return offsets[name](needs_int())
    if name in ("activeCycle", "completedCycles", "upcomingCycles"):
        raise PqlError(f"{name}() needs server-side PQL; {NEEDS_SERVER['cycle']}")
    if name in ("membersOf", "workspaceMembers"):
        raise PqlError(f"{name}() is not supported on Community Edition; pass member UUIDs instead")
    raise PqlError(f"unknown function {name}()")


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

# kind: enum (case-insensitive string), str, text, uuid, uuids (list), date, datetime, bool
FIELDS: dict[str, tuple[str, Callable[[Row, Context], Any]]] = {
    "priority": ("enum", lambda r, c: r.get("priority")),
    "stateGroup": ("enum", lambda r, c: c.state_group(ref_id(r.get("state")) or "")),
    "state": ("uuid", lambda r, c: ref_id(r.get("state"))),
    "isDraft": ("bool", lambda r, c: bool(r.get("is_draft"))),
    "isArchived": ("bool", lambda r, c: r.get("archived_at") is not None),
    "title": ("text", lambda r, c: r.get("name")),
    "text": (
        "text",
        lambda r, c: " ".join(
            filter(None, [r.get("name"), r.get("description_stripped") or _strip(r.get("description_html"))])
        ),
    ),
    "id": ("str", lambda r, c: c.identifier(r)),
    "dueDate": ("date", lambda r, c: r.get("target_date")),
    "startDate": ("date", lambda r, c: r.get("start_date")),
    "createdAt": ("datetime", lambda r, c: r.get("created_at")),
    "updatedAt": ("datetime", lambda r, c: r.get("updated_at")),
    "assignee": ("uuids", lambda r, c: ref_ids(r.get("assignees"))),
    "label": ("uuids", lambda r, c: ref_ids(r.get("labels"))),
    "project": ("uuid", lambda r, c: ref_id(r.get("project"))),
    "createdBy": ("uuid", lambda r, c: ref_id(r.get("created_by"))),
    "type": ("uuid", lambda r, c: ref_id(r.get("type_id")) or ref_id(r.get("type"))),
}


def _strip(html: str | None) -> str:
    return re.sub(r"<[^>]+>", " ", html or "")


def _as_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    try:
        if len(text) == 10:
            return date.fromisoformat(text)
        return _as_date(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError as exc:
        raise PqlError(f"{value!r} is not a date (use YYYY-MM-DD or a date function)") from exc


def _as_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PqlError(f"{value!r} is not a date (use YYYY-MM-DD or a date function)") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _comparable(kind: str, actual: Any, expected: Any) -> tuple[Any, Any]:
    """Bring a field value and a literal onto one scale for <, >, BETWEEN."""
    if kind == "date" or (kind == "datetime" and not isinstance(expected, datetime)):
        # A day-granular bound against a timestamp compares whole days.
        return _as_date(actual), _as_date(expected)
    if kind == "datetime":
        return _as_datetime(actual), _as_datetime(expected)
    return actual, expected


def _equal(kind: str, actual: Any, expected: Any) -> bool:
    if kind in ("date", "datetime"):
        return _as_date(actual) == _as_date(expected) if actual is not None else False
    if kind == "bool":
        return bool(actual) is bool(expected)
    if kind in ("enum", "str", "text") and isinstance(actual, str) and isinstance(expected, str):
        return actual.lower() == expected.lower()
    return actual == expected


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == []


def _compare(kind: str, op: str, actual: Any, expected: Any) -> bool:  # noqa: PLR0911
    if kind == "uuids":
        if op == "=":
            return expected in actual
        if op == "!=":
            return expected not in actual
        raise PqlError(f"operator {op} does not apply to a list field")
    if op == "=":
        return _equal(kind, actual, expected)
    if op == "!=":
        return not _equal(kind, actual, expected)
    if op == "~":
        if actual is None:
            return False
        needle = str(expected).lower()
        return str(actual).lower().startswith(needle) if kind == "str" else needle in str(actual).lower()
    if actual is None:
        return False
    left, right = _comparable(kind, actual, expected)
    if left is None or right is None:
        return False
    try:
        return {">": left > right, ">=": left >= right, "<": left < right, "<=": left <= right}[op]
    except TypeError as exc:
        raise PqlError(f"cannot compare {actual!r} with {expected!r}") from exc


# ---------------------------------------------------------------------------
# Predicate functions
# ---------------------------------------------------------------------------


def _predicate_function(name: str, args: list[Any], ctx: Context) -> Predicate:  # noqa: PLR0911
    def group(row: Row) -> str | None:
        return ctx.state_group(ref_id(row.get("state")) or "")

    standalone: dict[str, Predicate] = {
        "isOverdue": lambda r: (
            bool(r.get("target_date")) and _as_date(r["target_date"]) < ctx.today and group(r) in OPEN_STATES
        ),
        "hasNoAssignee": lambda r: not ref_ids(r.get("assignees")),
        "hasNoLabel": lambda r: not ref_ids(r.get("labels")),
        "isTopLevel": lambda r: not r.get("parent"),
        "isSubWorkItem": lambda r: bool(r.get("parent")),
        "hasStartAndDueDates": lambda r: bool(r.get("start_date")) and bool(r.get("target_date")),
        "isDraft": lambda r: bool(r.get("is_draft")),
        "isArchived": lambda r: r.get("archived_at") is not None,
    }
    if name in standalone:
        if args:
            raise PqlError(f"{name}() takes no arguments")
        return standalone[name]
    if name == "hasChildren":
        parents = ctx.parents()
        return lambda r: r.get("id") in parents
    if name in ("childOf", "parentOf"):
        if len(args) != 1 or not isinstance(args[0], str):
            raise PqlError(f'{name}() takes one work item id, e.g. {name}("HW-5")')
        target = ctx.resolve(args[0])
        if name == "childOf":
            return lambda r: ref_id(r.get("parent")) == target
        parent = next((ref_id(r.get("parent")) for r in ctx.rows if r.get("id") == target), None)
        return lambda r: parent is not None and r.get("id") == parent
    if name in ("linkedTo", "blockedBy", "blocks", "duplicateOf"):
        raise PqlError(f"{name}() is not supported on Community Edition; use `workitem_relation list` instead")
    if name in ("isEpic", "isIntake"):
        raise PqlError(f"{name}() is not supported on Community Edition")
    raise PqlError(f"{name}() is not a predicate; compare a field with it instead (e.g. dueDate < {name}())")


# ---------------------------------------------------------------------------
# Parser: OR < AND < NOT < comparison
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[Token], ctx: Context) -> None:
        self.tokens = tokens
        self.pos = 0
        self.ctx = ctx

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def take(self) -> Token:
        token = self.peek()
        self.pos += 1
        return token

    def accept(self, kind: str, value: Any = None) -> bool:
        token = self.peek()
        if token.kind == kind and (value is None or token.value == value):
            self.pos += 1
            return True
        return False

    def expect(self, kind: str, value: Any = None) -> Token:
        token = self.peek()
        if token.kind != kind or (value is not None and token.value != value):
            wanted = value or kind
            got = token.value if token.kind != "end" else "end of query"
            raise PqlError(f"expected {wanted!r}, got {got!r}")
        return self.take()

    def parse(self) -> Predicate:
        predicate = self.or_expr()
        if self.peek().kind != "end":
            raise PqlError(f"unexpected {self.peek().value!r}")
        return predicate

    def or_expr(self) -> Predicate:
        terms = [self.and_expr()]
        while self.accept("kw", "OR"):
            terms.append(self.and_expr())
        return terms[0] if len(terms) == 1 else (lambda r: any(t(r) for t in terms))

    def and_expr(self) -> Predicate:
        terms = [self.not_expr()]
        while self.accept("kw", "AND"):
            terms.append(self.not_expr())
        return terms[0] if len(terms) == 1 else (lambda r: all(t(r) for t in terms))

    def not_expr(self) -> Predicate:
        if self.accept("kw", "NOT"):
            inner = self.not_expr()
            return lambda r: not inner(r)
        return self.atom()

    def atom(self) -> Predicate:
        if self.accept("punct", "("):
            inner = self.or_expr()
            self.expect("punct", ")")
            return inner
        token = self.expect("word")
        name = token.value
        if name == "cf" and self.peek().value == "[":
            raise PqlError("custom properties (cf[...]) are not supported on Community Edition")
        if self.peek().value == "(" and self.peek().kind == "punct":
            return _predicate_function(name, self.arguments(), self.ctx)
        return self.comparison(name)

    def arguments(self) -> list[Any]:
        self.expect("punct", "(")
        args: list[Any] = []
        if not self.accept("punct", ")"):
            args.append(self.value())
            while self.accept("punct", ","):
                args.append(self.value())
            self.expect("punct", ")")
        return args

    def value(self) -> Any:
        token = self.take()
        if token.kind in ("string", "number"):
            return token.value
        if token.kind == "kw" and token.value in ("TRUE", "FALSE"):
            return token.value == "TRUE"
        if token.kind == "word" and self.peek().value == "(":
            return _value_function(token.value, self.arguments(), self.ctx)
        raise PqlError(f"expected a value, got {token.value!r}")

    def value_list(self) -> list[Any]:
        if self.peek().kind == "punct" and self.peek().value == "(":
            values = self.arguments()
        else:
            values = self.value()
        values = values if isinstance(values, list) else [values]
        return values

    def comparison(self, name: str) -> Predicate:  # noqa: PLR0911, PLR0912
        if name in NEEDS_SERVER:
            raise PqlError(f"`{name}` filters need server-side PQL; {NEEDS_SERVER[name]}")
        if name not in FIELDS:
            raise PqlError(f"unknown field {name!r}; supported on Community Edition: {', '.join(FIELDS)}")
        kind, getter = FIELDS[name]
        ctx = self.ctx

        def get(row: Row) -> Any:
            return getter(row, ctx)

        if self.accept("kw", "IS"):
            negate = self.accept("kw", "NOT")
            if not (self.accept("kw", "NULL") or self.accept("kw", "EMPTY")):
                raise PqlError("expected NULL or EMPTY after IS")
            return (lambda r: not _empty(get(r))) if negate else (lambda r: _empty(get(r)))

        negate = False
        if self.peek().kind == "kw" and self.peek().value == "NOT":
            self.take()
            negate = True
            if self.peek().value != "IN":
                raise PqlError("expected IN after NOT")
        if self.accept("kw", "IN"):
            wanted = self.value_list()

            def contained(row: Row) -> bool:
                actual = get(row)
                if kind == "uuids":
                    return any(value in actual for value in wanted)
                return any(_equal(kind, actual, value) for value in wanted)

            return (lambda r: not contained(r)) if negate else contained

        if self.accept("kw", "BETWEEN"):
            if self.peek().value == "(" and self.peek().kind == "punct":
                bounds = self.arguments()
                if len(bounds) != 2:
                    raise PqlError("BETWEEN takes two values: BETWEEN (low, high)")
                low, high = bounds
            else:
                low = self.value()
                self.expect("kw", "AND")
                high = self.value()
            return lambda r: _compare(kind, ">=", get(r), low) and _compare(kind, "<=", get(r), high)

        op = self.expect("op").value
        expected = self.value()
        if isinstance(expected, list):
            raise PqlError(f"{name} {op} takes one value; use IN for a list")
        if kind == "enum" and isinstance(expected, str):
            expected = expected.lower()
        _compare(kind, op, None if kind != "uuids" else [], expected)  # reject bad operators up front
        return lambda r: _compare(kind, op, get(r), expected)


def compile_pql(text: str, ctx: Context) -> Predicate:
    """A predicate over dumped work item rows, or PqlError explaining why not."""
    if not text.strip():
        return lambda _row: True
    return _Parser(tokenize(text), ctx).parse()
