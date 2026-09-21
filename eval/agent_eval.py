"""Layered evaluation of the product's only chat path: ``/agent/chat``.

Runs the real runtime in-process (``src.api.agent_routes.build_runtime``) so the
trajectory layer can see rounds, step timing and dropped citations -- things an
HTTP client cannot observe -- while an optional preflight still checks the HTTP
surface for drift.  Retrieval-only measurement lives in
``src/evaluation/retrieval_probe.py`` (L1); this module is L2/L3/L4.

Run inside the api image (needs chromadb / sentence-transformers / reranker):

    docker compose run --rm --no-deps \\
      -v "<repo>/eval:/app/eval" -v "<repo>/test_data:/app/test_data" \\
      -v "<repo>/data:/app/data" -v "<repo>/config:/app/config" -v "<repo>/tmp:/app/tmp" \\
      api python -m eval.agent_eval --collection <id> --out tmp/baselines/agent_eval.jsonl

Writes only to ``--out`` and its sibling work directory; ``data/`` is refused
outright, because that is where the index under test lives.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

from eval import outcomes as O
from eval import report as R

MODE_DEEP = "deep"
DEFAULT_QUESTIONS = "test_data/evaluation_questions.jsonl"
DEFAULT_COLLECTION = "col_0d83015737fd"
ANSWER_CHARS = 4000


# ------------------------------------------------------------------- repo facts
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_paths(value: str) -> list[Path]:
    root = repo_root()
    paths: list[Path] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        paths.append(path if path.is_absolute() else root / path)
    return paths


def read_index_fingerprint(index_dir: Path) -> dict:
    """Which index the run actually measured -- without it a diff is unattributable."""

    fingerprint: dict = {"index_dir": str(index_dir), "version_id": None, "chunk_count": None}
    current = index_dir / "current.json"
    if not current.exists():
        fingerprint["error"] = "no current.json (index not published?)"
        return fingerprint
    version_id = json.loads(current.read_text(encoding="utf-8")).get("version_id")
    fingerprint["version_id"] = version_id
    manifest_path = index_dir / "versions" / str(version_id) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprint.update({
            "chunk_count": manifest.get("chunk_count"),
            "chunker_version": manifest.get("chunker_version"),
            "element_schema_version": manifest.get("element_schema_version"),
            "embedding_model": manifest.get("embedding_model"),
            "embedding_dimension": manifest.get("embedding_dimension"),
        })
    return fingerprint


def load_chunk_map(parsed_dir: Path) -> dict[str, dict]:
    """chunk_id -> (filename, page) from the parsed truth, not from the model."""

    index: dict[str, dict] = {}
    for path in sorted(parsed_dir.glob("*/chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = str(row.get("chunk_id") or "")
            if not chunk_id:
                continue
            index[chunk_id] = {
                "filename": row.get("filename") or "",
                "page": row.get("page"),
                # A3: last page of the chunk (None on pre-upgrade artifacts).
                "page_end": row.get("page_end"),
                "document_id": row.get("document_id") or "",
                "chunk_type": row.get("chunk_type") or "",
            }
    return index


def load_registry(registry_path: Path) -> dict[str, str]:
    if not registry_path.exists():
        return {}
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    records = payload.get("documents") if isinstance(payload, dict) else payload
    mapping: dict[str, str] = {}
    for record in records or []:
        if isinstance(record, dict) and record.get("document_id"):
            mapping[str(record["document_id"])] = str(record.get("original_filename") or "")
    return mapping


def collection_document_ids(db_path: Path, collection_id: str) -> list[str]:
    if not db_path.exists():
        return []
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            "SELECT document_id FROM collection_documents WHERE collection_id = ?",
            (collection_id,),
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


# ------------------------------------------------------------------- isolation
def clone_agent_db(source: Path, work_dir: Path, *, reuse: bool = False) -> Path:
    """Copy the product DB so eval sessions never accumulate in it.

    ``reuse=True`` (a ``--resume`` regeneration) keeps the existing clone: the
    rows were already recorded, and re-copying would discard the session trail
    that belongs to that run.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / "app.db"
    if reuse and target.exists():
        return target
    if source.exists():
        shutil.copy2(source, target)
    return target


# ------------------------------------------------------------------------ rows
def normalize_trace(tool_trace: list[dict] | None) -> list[dict]:
    steps: list[dict] = []
    for step in tool_trace or []:
        route = step.get("route")
        route = route if isinstance(route, dict) else {}
        steps.append({
            "round": step.get("round"),
            "tool": step.get("tool"),
            "arguments": step.get("arguments") or {},
            "ok": step.get("ok"),
            "error": step.get("error"),
            "track": route.get("track"),
            "provider": route.get("provider"),
            "observed": step.get("observed") or [],
            "offset_s": step.get("offset_s"),
            "duration_s": step.get("duration_s"),
        })
    return steps


def _covers_page(citation: dict, expected_pages: set) -> bool:
    """A3: does this citation cover any expected page?

    A chunk that straddles a page break covers every page in ``page``..``page_end``
    -- a passage starting on p3 and running onto p4 is legitimate evidence for a
    question whose answer sits on p4.
    """

    page = citation.get("page")
    if page is None:
        return False
    end = citation.get("page_end") or page
    try:
        low, high = int(page), int(end)
    except (TypeError, ValueError):
        return False
    for want in expected_pages:
        try:
            if low <= int(want) <= high:
                return True
        except (TypeError, ValueError):
            continue
    return False


def build_row(*, sample: dict, result, elapsed_s: float, chunk_map: dict[str, dict],
              docs: dict[str, str], max_steps: int, mode: str) -> dict:
    expect = O.resolve_expect(sample)
    expected_docs = set(sample.get("expected_documents") or [])
    expected_pages = set(sample.get("expected_pages") or [])

    citations: list[dict] = []
    for item in result.citations or []:
        chunk_id = str(item.get("chunk_id") or "")
        truth = chunk_map.get(chunk_id) or {}
        model_page = item.get("page")
        truth_page = truth.get("page")
        truth_page_end = truth.get("page_end") or truth_page
        # D59 (2026-09-21): a chunk can span pages -- A3 added ``page_end`` for exactly
        # that -- so a citation naming ANY page in [page, page_end] is faithful.
        # Comparing only against the start page flagged a legitimate page-5 citation of
        # a 4-5 chunk (row #12, sentence-bert.pdf) as a mismatch.
        if model_page is not None and truth_page is not None and truth_page_end is not None:
            page_fidelity = bool(truth_page <= model_page <= truth_page_end)
        else:
            page_fidelity = None
        citations.append({
            "chunk_id": chunk_id,
            "filename": truth.get("filename") or item.get("filename") or docs.get(str(item.get("document_id") or ""), ""),
            "page": truth_page if truth_page is not None else model_page,
            "page_end": truth_page_end,
            "model_page": model_page,
            "page_fidelity": page_fidelity,
            "chunk_type": truth.get("chunk_type") or "",
            "quote": str(item.get("quote") or "")[:400],
        })

    hit = any(
        citation["filename"] in expected_docs and _covers_page(citation, expected_pages)
        for citation in citations
    )
    coverage, missing = O.number_coverage(sample.get("reference_answer") or "", result.answer or "")
    trace = normalize_trace(result.tool_trace)
    rounds = list(getattr(result, "rounds", []) or [])
    dropped = list(getattr(result, "dropped_citations", []) or [])
    rounds_total = len(rounds) if rounds else len({step["round"] for step in trace if step.get("round")})

    return {
        "index": sample["index"],
        "question": sample["question"],
        "expect": expect,
        "answerable": bool(sample.get("answerable", True)),
        "expected_documents": sorted(expected_docs),
        "expected_pages": sorted(expected_pages),
        "reference_answer": sample.get("reference_answer") or "",
        "mode": mode,
        "max_steps": max_steps,
        "status": "ok",
        "error": None,
        "outcome": O.classify({
            "status": "ok",
            "failure": result.failure,
            "expect": expect,
            "answer": result.answer,
            "refused": result.refused,
            "citations": citations,
            "citation_hit": hit,
        }),
        "refused": bool(result.refused),
        "failure": result.failure,
        "gate": result.gate,
        "warnings": list(result.warnings or []),
        "elapsed_s": round(elapsed_s, 2),
        "answer": (result.answer or "")[:ANSWER_CHARS],
        "answer_chars": len(result.answer or ""),
        "truncated_answer": len(result.answer or "") > ANSWER_CHARS,
        "citations": citations,
        "citation_hit": hit,
        "citation_count": len(citations),
        "citation_dropped": len(dropped),
        "citation_dropped_ids": dropped[:20],
        "number_coverage": coverage,
        "numbers_missing": missing,
        "trace": trace,
        "rounds": rounds,
        "rounds_total": rounds_total,
        "tool_calls_total": len(trace),
        "timing_available": any(step.get("duration_s") is not None for step in trace),
        "read_by_page": any(
            step.get("tool") == "read" and not (step.get("arguments") or {}).get("chunk_id")
            for step in trace
        ),
    }


def transport_row(sample: dict, error: str, max_steps: int, mode: str) -> dict:
    return {
        "index": sample["index"],
        "question": sample["question"],
        "expect": O.resolve_expect(sample),
        "answerable": bool(sample.get("answerable", True)),
        "expected_documents": sample.get("expected_documents") or [],
        "expected_pages": sample.get("expected_pages") or [],
        "reference_answer": sample.get("reference_answer") or "",
        "mode": mode,
        "max_steps": max_steps,
        "status": "transport_error",
        "error": error,
        "outcome": O.OUTCOME_TRANSPORT_ERROR,
        "refused": False,
        "failure": None,
        "gate": {},
        "warnings": [],
        "elapsed_s": 0.0,
        "answer": "",
        "answer_chars": 0,
        "truncated_answer": False,
        "citations": [],
        "citation_hit": False,
        "citation_count": 0,
        "citation_dropped": 0,
        "citation_dropped_ids": [],
        "number_coverage": None,
        "numbers_missing": [],
        "trace": [],
        "rounds": [],
        "rounds_total": 0,
        "tool_calls_total": 0,
        "timing_available": False,
        "read_by_page": False,
    }


# ------------------------------------------------------------------- preflight
def check_http_surface(url: str) -> dict:
    import urllib.error
    import urllib.request

    result: dict = {"url": url, "reachable": False}
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/openapi.json", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        paths = sorted(payload.get("paths") or {})
        result.update({"reachable": True, "has_agent_chat": "/agent/chat" in paths, "paths": paths})
    except Exception as error:  # noqa: BLE001 - drift check must not abort the run
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def meta_block(args, samples: list[dict], fingerprint: dict, chunk_map: dict[str, dict], docs: dict[str, str]) -> dict:
    from src.config.settings import get_agent_settings, get_retrieval_settings

    from src.llm import ModelRegistry

    agent_settings = get_agent_settings()
    retrieval = get_retrieval_settings()
    try:
        models = ModelRegistry.load(agent_settings.models_config_path or None)
        model_names = {
            role: getattr(models.resolve(role), "model", None) for role in ("agent", "router", "summarizer")
        }
    except Exception as error:  # noqa: BLE001 - recorded, not fatal
        model_names = {"error": f"{type(error).__name__}: {error}"}
    return {
        "harness_version": O.HARNESS_VERSION,
        "chain": "agent",
        "started_at": datetime.now().astimezone().isoformat(),
        "question_files": [str(path) for path in parse_paths(args.questions)],
        "questions": len(samples),
        "collection_id": args.collection,
        "collection_documents": sorted(docs.values()),
        "collection_document_ids": sorted(docs.keys()),
        "index": fingerprint,
        "chunk_map_size": len(chunk_map),
        "chunk_map_matches_manifest": (
            fingerprint.get("chunk_count") == len(chunk_map) if fingerprint.get("chunk_count") else None
        ),
        "models": model_names,
        "settings": {
            "AGENT_MAX_STEPS": agent_settings.max_steps,
            "AGENT_MAX_GATE_BLOCKS": agent_settings.max_gate_blocks,
            "AGENT_ROUTE_MODE": agent_settings.route_mode,
            "tool_max_hits": agent_settings.tool_max_hits,
            "tool_text_chars": agent_settings.tool_text_chars,
            "RETRIEVAL_MIN_SCORE": retrieval.min_score,
            "RETRIEVAL_FINAL_K": retrieval.final_k,
            "RERANK_ENABLED": retrieval.rerank_enabled,
            "RERANK_MODEL": retrieval.rerank_model,
            "HF_HUB_OFFLINE": os.getenv("HF_HUB_OFFLINE", ""),
            "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", ""),
        },
    }


# ------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L2/L3/L4 evaluation of the agent path.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS,
                        help="comma-separated JSONL question files (core set first: indices stay stable)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--mode", default=MODE_DEEP, choices=[MODE_DEEP, "chat"])
    parser.add_argument("--out", required=True, help="JSONL output path (never inside data/)")
    parser.add_argument("--limit", type=int, default=0, help="first N questions only")
    parser.add_argument("--indices", default="", help="comma-separated question indices to run")
    parser.add_argument("--only", default="", choices=["", "answerable", "unanswerable", "deny", "refuse"])
    parser.add_argument("--resume", action="store_true", help="skip rows already present in --out for this fingerprint")
    parser.add_argument("--agent-db", default="data/app.db")
    parser.add_argument("--parsed-dir", default="data/parsed")
    parser.add_argument("--registry", default="data/registry.json")
    parser.add_argument("--index-dir", default="data/chroma")
    parser.add_argument("--http-check", default="", help="optional URL whose /openapi.json must expose /agent/chat")
    parser.add_argument("--baseline", default="", help="path to a previous run's .summary.json")
    parser.add_argument(
        "--audit-verdicts",
        default="",
        help="JSON mapping question index -> {verdict: correct|hallucinated|other, note}: the reviewed结论 for audit rows",
    )
    parser.add_argument("--preflight-only", action="store_true", help="run L0 checks and stop")
    parser.add_argument("--no-isolate-db", action="store_true", help="use the product DB directly (not recommended)")
    return parser.parse_args(argv)


def select_samples(samples: list[dict], args) -> list[dict]:
    selected = samples
    if args.indices.strip():
        wanted = {int(item) for item in args.indices.split(",") if item.strip()}
        selected = [sample for sample in selected if sample["index"] in wanted]
    if args.only:
        mapping = {"answerable": O.EXPECT_ANSWER, "unanswerable": None, "deny": O.EXPECT_DENY, "refuse": O.EXPECT_REFUSE}
        target = mapping[args.only]
        if target is None:
            selected = [sample for sample in selected if O.resolve_expect(sample) != O.EXPECT_ANSWER]
        else:
            selected = [sample for sample in selected if O.resolve_expect(sample) == target]
    if args.limit:
        selected = selected[: args.limit]
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    out_path = O.assert_writable_output(args.out, repo_root=root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = O.assert_writable_output(out_path.parent / f"{out_path.stem}.work", repo_root=root)

    samples = O.load_samples(parse_paths(args.questions))
    fingerprint = read_index_fingerprint(root / args.index_dir)
    chunk_map = load_chunk_map(root / args.parsed_dir)
    registry = load_registry(root / args.registry)
    agent_db = root / args.agent_db
    document_ids = collection_document_ids(agent_db, args.collection)
    docs = {document_id: registry.get(document_id, "") for document_id in document_ids}

    problems: list[str] = []
    if not chunk_map:
        problems.append(f"no chunk map under {root / args.parsed_dir} (citation (file,page) cannot be resolved)")
    if not document_ids:
        problems.append(f"collection {args.collection} has no documents in {agent_db}")
    missing_docs = {
        name for sample in samples for name in (sample.get("expected_documents") or [])
        if name not in set(docs.values())
    }
    if missing_docs:
        problems.append(f"question set expects documents missing from the collection: {sorted(missing_docs)}")
    http_result = check_http_surface(args.http_check) if args.http_check else {"skipped": True}
    if http_result.get("reachable") and not http_result.get("has_agent_chat"):
        problems.append("HTTP surface is reachable but /agent/chat is not registered")

    print(f"questions={len(samples)} chunk_map={len(chunk_map)} documents={len(docs)} "
          f"index={fingerprint.get('version_id')} chunks={fingerprint.get('chunk_count')}", flush=True)
    if http_result.get("reachable"):
        print(f"http surface ok: /agent/chat present at {http_result['url']}", flush=True)
    elif args.http_check:
        print(f"http surface not reachable ({http_result.get('error')}) — routes not verified this run", flush=True)
    for problem in problems:
        print(f"PREFLIGHT: {problem}", flush=True)

    if args.preflight_only:
        return 1 if problems else 0
    if problems:
        print("aborting: fix the preflight problems (or use --preflight-only to inspect)", flush=True)
        return 1

    # --- isolation -----------------------------------------------------------------
    if args.no_isolate_db:
        os.environ.pop("AGENT_DB_PATH", None)
    else:
        clone = clone_agent_db(agent_db, work_dir, reuse=args.resume)
        os.environ["AGENT_DB_PATH"] = str(clone)
        print(f"isolated agent db: {clone}", flush=True)

    from src.api.agent_routes import build_runtime

    runtime, db, _, _ = build_runtime()
    max_steps = int(getattr(runtime.settings, "max_steps", 0) or 0)
    todo = select_samples(samples, args)

    existing: list[dict] = []
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(json.loads(line))
        print(f"resume: {len(existing)} rows already recorded", flush=True)
    else:
        out_path.write_text("", encoding="utf-8")
    done = {(row.get("index"), row.get("question")) for row in existing}

    rows = list(existing)
    for position, sample in enumerate(todo, start=1):
        if (sample["index"], sample["question"]) in done:
            print(f"[{position}/{len(todo)}] #{sample['index']} already recorded, skipped", flush=True)
            continue
        print(f"\n[{position}/{len(todo)}] #{sample['index']} [{O.resolve_expect(sample)}] {sample['question'][:60]}",
              flush=True)
        started = time.perf_counter()
        try:
            session_id = db.create_session(collection_id=args.collection, mode=args.mode)
            result = runtime.run(session_id, sample["question"], args.mode)
            row = build_row(sample=sample, result=result, elapsed_s=time.perf_counter() - started,
                            chunk_map=chunk_map, docs=docs, max_steps=max_steps, mode=args.mode)
        except Exception as error:  # noqa: BLE001 - one bad turn must not kill the run
            row = transport_row(sample, f"{type(error).__name__}: {error}", max_steps, args.mode)
        rows.append(row)
        with out_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {row['elapsed_s']}s outcome={row['outcome']} failure={row['failure']} "
              f"cites={row['citation_count']} dropped={row['citation_dropped']} "
              f"rounds={row['rounds_total']} calls={row['tool_calls_total']} "
              f"numcov={row['number_coverage']}", flush=True)

    verdicts = O.load_verdicts(parse_paths(args.audit_verdicts)[0]) if args.audit_verdicts else None
    summary = O.summarize(rows, verdicts=verdicts)
    summary["meta"] = meta_block(args, samples, fingerprint, chunk_map, docs)
    summary["meta"]["http_check"] = http_result
    summary["meta"]["audit_verdicts_file"] = str(args.audit_verdicts or "")
    summary["selection"] = {
        "indices_run_now": [sample["index"] for sample in todo],
        "only": args.only,
        "limit": args.limit,
        "rows_in_file": len(rows),
        "note": "summary 聚合的是 --out 里的全部行，不是本次新跑的行",
    }
    summary_path = out_path.parent / f"{out_path.stem}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path = out_path.parent / f"{out_path.stem}.audit.md"
    audit_path.write_text(R.audit_markdown(rows, summary), encoding="utf-8")

    print(R.render_text(summary, rows), flush=True)
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        print(R.comparison(baseline, summary), flush=True)
        base_index = ((baseline.get("meta") or {}).get("index") or {}).get("version_id")
        current_index = fingerprint.get("version_id")
        if base_index and base_index != current_index:
            print(f"  ! index differs: {base_index} -> {current_index} (differences are not code-attributable)", flush=True)
    print(f"\nrows  -> {out_path}\nsummary -> {summary_path}\naudit -> {audit_path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())