"""Tests for the D8/D11 parse-quality payload behind the library badge.

The high-value case here is :func:`test_dropped_by_role_matches_the_chunker_exactly`:
``document_quality._dropped_by_role`` claims to mirror the chunker's drop rule, and
when those two drift apart the badge reports numbers that never happened.  That is
not hypothetical -- D17 renamed the policy on the producer side only, and the two
rules disagreed until it was caught by a repo-wide sweep.
"""

from __future__ import annotations

import json
import pathlib

from src.api.document_quality import _dropped_by_role, document_quality
from src.ingestion.ordered_chunker import chunk_ordered_document


def _element(element_id: str, *, role: str = "body", policy: str = "embed", notice: str | None = None) -> dict:
    return {
        "element_id": element_id,
        "ordinal": int(element_id.strip("e") or 1),
        "type": "paragraph",
        "role": role,
        "index_policy": policy,
        "page": 1,
        "heading_path_norm": ["Section"],
        "text": f"text of {element_id}",
        "search_text": f"text of {element_id}",
        "structure": {},
        "atomic": False,
        "parse_status": "parsed",
        "payload": {},
        **({"user_notice": notice} if notice else {}),
    }


def _ordered(elements: list[dict]) -> dict:
    return {"document_id": "doc_test", "sequence": elements, "source_pdf": "test.pdf"}


def _write_artifacts(root: pathlib.Path, document_id: str, **files: object) -> pathlib.Path:
    directory = root / document_id
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (directory / f"{name}.json").write_text(
            payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
        )
    return directory


# ------------------------------------------------------- the mirror invariant
def test_dropped_by_role_matches_the_chunker_exactly():
    """Whatever the chunker drops, this rule must count -- same roles, same policies.

    A divergence here is a silent lie in the UI: the badge would claim elements were
    dropped while the chunker embedded them (or the reverse).  D17 produced exactly
    that divergence by renaming ``metadata_only`` to ``skip`` on one side only, so
    every policy name is exercised here, legacy spelling included.
    """

    ordered = _ordered(
        [
            _element("e1", role="body", policy="embed"),
            _element("e2", role="body", policy="skip"),
            _element("e3", role="body", policy="metadata_only"),  # legacy, pre-D17 artifacts
            _element("e4", role="body", policy="drop"),
            _element("e5", role="reference", policy="drop"),
            _element("e6", role="preamble", policy="drop"),
            _element("e7", role="noise", policy="drop"),
            _element("e8", role="footnote", policy="skip"),
            _element("e9", role="body", policy="embed"),
        ]
    )

    _chunks, report = chunk_ordered_document(ordered, filename="test.pdf")

    assert _dropped_by_role(ordered) == report["dropped_by_role"]
    # And the rule really is doing something -- not two empty dicts agreeing.
    assert sum(report["dropped_by_role"].values()) == 7


def test_every_policy_the_refiner_can_produce_is_covered():
    """``assign_role`` only ever emits embed/drop/skip -- all three must be handled."""

    ordered = _ordered([_element(f"e{i}", policy=policy) for i, policy in enumerate(("embed", "drop", "skip"), 1)])
    _chunks, report = chunk_ordered_document(ordered, filename="test.pdf")
    assert report["dropped_by_role"] == {"body": 2}
    assert _dropped_by_role(ordered) == {"body": 2}


# ------------------------------------------------------- the public payload
def test_no_artifacts_reports_unknown_rather_than_zero(tmp_path):
    """"We know nothing" must be distinguishable from "nothing was dropped"."""

    payload = document_quality("doc_missing", tmp_path)
    assert payload == {"degraded": False, "user_notice": None, "quality": None}


def test_corrupt_artifacts_never_raise(tmp_path):
    """A half-written parse directory must not 500 the library listing."""

    _write_artifacts(tmp_path, "doc_bad", ordered="{not json", quality_report="[[[", chunk_report="")
    payload = document_quality("doc_bad", tmp_path)
    assert set(payload) == {"degraded", "user_notice", "quality"}


def test_unparsed_tables_mark_the_document_degraded(tmp_path):
    _write_artifacts(
        tmp_path,
        "doc_deg",
        quality_report={"quality": {"table_unparsed": 2, "formula_unparsed": 1}},
        ordered=_ordered([_element("e1")]),
    )
    payload = document_quality("doc_deg", tmp_path)
    assert payload["degraded"] is True
    assert payload["quality"] is not None
    assert payload["quality"]["parse_quality"]


def test_user_notice_is_the_most_common_one(tmp_path):
    """A document with six degraded tables must not lead with a one-off paragraph."""

    _write_artifacts(
        tmp_path,
        "doc_notice",
        ordered=_ordered(
            [
                _element("e1", notice="表格解析不完整"),
                _element("e2", notice="公式解析不完整"),
                _element("e3", notice="表格解析不完整"),
            ]
        ),
    )
    payload = document_quality("doc_notice", tmp_path)
    assert payload["user_notice"] == "表格解析不完整"
    assert payload["degraded"] is True


def test_dropped_total_adds_the_byline_demotion_from_the_chunk_report(tmp_path):
    """The byline demotion is only knowable from the chunker's report, not from ordered.json."""

    _write_artifacts(
        tmp_path,
        "doc_byline",
        ordered=_ordered([_element("e1", policy="skip"), _element("e2")]),
        chunk_report={"demoted_byline": 3},
    )
    dropped = document_quality("doc_byline", tmp_path)["quality"]["dropped_elements"]
    assert dropped["by_role"]["byline"] == 3
    assert dropped["total"] == sum(dropped["by_role"].values()) == 4


def test_missing_sequence_reports_cannot_compute(tmp_path):
    """No sequence -> ``None``, so the UI does not read it as "zero dropped"."""

    _write_artifacts(tmp_path, "doc_noseq", ordered={"document_id": "doc_noseq"})
    payload = document_quality("doc_noseq", tmp_path)
    assert payload["quality"]["dropped_elements"] is None
