"""Plane Community Edition has none of the endpoints `workitem_relation` targets.

CE v1.4.x serves neither `/dependencies/`, `/custom-relations/` nor
`/work-item-relation-definitions/`; every call 404s. It does serve the older
`/work-items/<id>/relations/` (GET + POST, no delete). On a 404 the tool falls
back to that endpoint, or says plainly what CE cannot do. Delete uses
`/relations/remove/`, which only the homelab overlay on Plane's API adds
(Tomvis/homelab-stacks hosts/101/plane/api-overlay).
"""

from __future__ import annotations

import pytest
from plane.errors.errors import HttpError
from plane.models.work_items import CreateWorkItemRelation

NOT_FOUND = HttpError("Not Found", status_code=404, response={"error": "Page not found."})

CE_LISTING = {
    "blocking": [{"project_id": "p", "issue_id": "w-2"}],
    "blocked_by": [],
    "duplicate": [],
    "relates_to": [],
    "start_after": [],
    "start_before": [],
    "finish_after": [],
    "finish_before": [],
}


@pytest.fixture
def ce(spy):
    for path in (
        "work_items.dependencies.list",
        "work_items.dependencies.create",
        "work_items.dependencies.remove",
        "work_items.custom_relations.list",
        "work_items.custom_relations.create",
        "work_items.custom_relations.remove",
        "work_item_relation_definitions.list",
    ):
        spy.returns[path] = NOT_FOUND
    return spy


def _call(registered, **kwargs):
    return registered["workitem_relation"].fn(**kwargs)


def test_list_falls_back_to_the_relations_endpoint(registered, ce):
    ce.returns["work_items.relations._get"] = CE_LISTING

    result = _call(registered, action="list", project_id="p", workitem_id="w")

    assert result == {"relations": CE_LISTING}
    get = [c for c in ce.recorder.calls if c.method == "work_items.relations._get"]
    assert get[0].kwargs["endpoint"] == "acme/projects/p/work-items/w/relations"


@pytest.mark.parametrize("relation_type", ["blocked_by", "blocking", "start_before", "finish_after"])
def test_create_dependency_falls_back_to_the_relations_endpoint(registered, ce, relation_type):
    _call(
        registered, action="create", project_id="p", workitem_id="w", workitem_ids=["w-2"], relation_type=relation_type
    )

    assert ce.recorder.methods == ["work_items.dependencies.create", "work_items.relations.create"]
    data = ce.recorder.calls[-1].kwargs["data"]
    assert isinstance(data, CreateWorkItemRelation)
    assert (data.relation_type, data.issues) == (relation_type, ["w-2"])


@pytest.mark.parametrize("relation_type", ["relates_to", "duplicate"])
def test_ce_only_types_go_straight_to_the_relations_endpoint(registered, ce, relation_type):
    _call(
        registered, action="create", project_id="p", workitem_id="w", workitem_ids=["w-2"], relation_type=relation_type
    )

    assert ce.recorder.methods == ["work_items.relations.create"]
    assert ce.recorder.only().kwargs["data"].relation_type == relation_type


def test_custom_relation_on_ce_is_refused_with_a_reason(registered, ce):
    result = _call(
        registered,
        action="create",
        project_id="p",
        workitem_id="w",
        workitem_ids=["w-2"],
        relation_definition_id="d",
        relation_definition_label="causes",
    )

    assert isinstance(result, str) and result.startswith("Error:")
    assert "Community" in result


@pytest.mark.parametrize("is_dependency", [True, False])
def test_delete_on_ce_falls_back_to_the_overlay_remove_endpoint(registered, ce, is_dependency):
    ce.returns["work_items.relations._post"] = None

    result = _call(
        registered,
        action="delete",
        project_id="p",
        workitem_id="w",
        related_workitem_id="w-2",
        is_dependency=is_dependency,
    )

    assert result is None
    post = ce.recorder.calls[-1]
    assert post.method == "work_items.relations._post"
    assert post.kwargs["endpoint"] == "acme/projects/p/work-items/w/relations/remove"
    assert post.kwargs["data"] == {"related_issue": "w-2"}


def test_delete_on_ce_without_the_overlay_is_refused_with_a_reason(registered, ce):
    ce.returns["work_items.relations._post"] = NOT_FOUND

    result = _call(registered, action="delete", project_id="p", workitem_id="w", related_workitem_id="w-2")

    assert isinstance(result, str) and result.startswith("Error:")
    assert "web UI" in result


def test_delete_on_ce_of_a_missing_relation_says_so(registered, ce):
    ce.returns["work_items.relations._post"] = HttpError(
        "Not Found", status_code=404, response={"error": "relation not found"}
    )

    result = _call(registered, action="delete", project_id="p", workitem_id="w", related_workitem_id="w-2")

    assert result == "Error: no relation between w and w-2."


def test_list_definitions_on_ce_lists_the_built_in_types(registered, ce):
    result = _call(registered, action="list_definitions")

    assert result["custom_definitions"] == []
    assert {"blocked_by", "relates_to", "duplicate"} <= set(result["built_in_dependencies"])


def test_other_errors_are_not_swallowed(registered, spy):
    spy.returns["work_items.dependencies.list"] = HttpError("Forbidden", status_code=403)

    with pytest.raises(HttpError):
        _call(registered, action="list", project_id="p", workitem_id="w")


def test_commercial_path_is_unchanged(registered, spy):
    _call(
        registered, action="create", project_id="p", workitem_id="w", workitem_ids=["w-2"], relation_type="blocked_by"
    )

    assert spy.recorder.methods == ["work_items.dependencies.create"]
