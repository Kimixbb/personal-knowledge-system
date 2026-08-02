from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from personal_rag.chunking import MultilingualChunker
from personal_rag.config import Settings, ensure_vault
from personal_rag.embeddings import Qwen3Embeddings
from personal_rag.manifest import Manifest
from personal_rag.retrieval import RetrievedPassage, Retriever
from personal_rag.synchronizer import Synchronizer
from personal_rag.vector_store import ChromaVectorStore


class BenchmarkSynchronizer(Protocol):
    def sync_library(self, *, raise_on_errors: bool = False) -> Any: ...


class BenchmarkRetriever(Protocol):
    def retrieve(self, question: str) -> list[RetrievedPassage]: ...


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    category: str
    question: str
    expected_paths: tuple[str, ...]
    expected_fact: str
    expected_language: str
    forbidden_paths: tuple[str, ...] = ()
    forbidden_text: tuple[str, ...] = ()
    setup: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    id: str
    category: str
    question: str
    expected_fact: str
    expected_language: str
    passed: bool | None
    expected_path_ranks: dict[str, int]
    missing_paths: tuple[str, ...]
    forbidden_paths_found: tuple[str, ...]
    forbidden_text_found: tuple[str, ...]
    retrieved_paths: tuple[str, ...]
    passages: tuple[dict[str, Any], ...]
    requires_manual_answer_review: bool = True


def _string_tuple(value: Any, field_name: str, case_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{case_id}: {field_name} must be a list of strings")
    return tuple(value)


def _parse_case(value: Any, line_number: int) -> EvaluationCase:
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: evaluation case must be an object")
    required = (
        "id",
        "category",
        "question",
        "expected_paths",
        "expected_fact",
        "expected_language",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"line {line_number}: missing fields: {', '.join(missing)}")
    case_id = str(value["id"])
    setup = value.get("setup")
    if setup is not None and (
        not isinstance(setup, dict)
        or not all(isinstance(key, str) and isinstance(item, str) for key, item in setup.items())
    ):
        raise ValueError(f"{case_id}: setup must be an object containing strings")
    return EvaluationCase(
        id=case_id,
        category=str(value["category"]),
        question=str(value["question"]),
        expected_paths=_string_tuple(value["expected_paths"], "expected_paths", case_id),
        expected_fact=str(value["expected_fact"]),
        expected_language=str(value["expected_language"]),
        forbidden_paths=_string_tuple(
            value.get("forbidden_paths", []), "forbidden_paths", case_id
        ),
        forbidden_text=_string_tuple(
            value.get("forbidden_text", []), "forbidden_text", case_id
        ),
        setup=setup,
    )


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            case = _parse_case(value, line_number)
            if case.id in seen_ids:
                raise ValueError(f"duplicate evaluation case id: {case.id}")
            seen_ids.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("evaluation set is empty")
    return cases


def _safe_vault_target(vault_root: Path, relative_path: str) -> Path:
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or not posix_path.parts or posix_path.parts[0] != "library":
        raise ValueError("evaluation setup path must be relative to library")
    root = vault_root.resolve()
    target = (root / Path(*posix_path.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("evaluation setup path escapes the benchmark vault") from exc
    return target


def apply_case_setup(case: EvaluationCase, vault_root: Path) -> None:
    if case.setup is None:
        return
    action = case.setup.get("action", "")
    relative_path = case.setup.get("path", "")
    target = _safe_vault_target(vault_root, relative_path)
    if action == "replace_text":
        old = case.setup.get("old", "")
        new = case.setup.get("new", "")
        text = target.read_text(encoding="utf-8")
        if not old or text.count(old) != 1:
            raise ValueError(
                f"{case.id}: setup text must occur exactly once in {relative_path}"
            )
        old_mtime = target.stat().st_mtime_ns
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        if target.stat().st_mtime_ns <= old_mtime:
            updated_mtime = old_mtime + 1_000_000
            os.utime(target, ns=(updated_mtime, updated_mtime))
        return
    if action == "delete":
        target.unlink()
        return
    raise ValueError(f"{case.id}: unsupported setup action: {action or '<empty>'}")


def evaluate_retrieval(
    case: EvaluationCase, passages: Sequence[RetrievedPassage]
) -> EvaluationResult:
    path_ranks: dict[str, int] = {}
    retrieved_paths: list[str] = []
    for passage in passages:
        if passage.relative_path not in path_ranks:
            path_ranks[passage.relative_path] = passage.rank
            retrieved_paths.append(passage.relative_path)

    expected_path_ranks = {
        path: path_ranks[path] for path in case.expected_paths if path in path_ranks
    }
    missing_paths = tuple(path for path in case.expected_paths if path not in path_ranks)
    forbidden_paths_found = tuple(
        path for path in case.forbidden_paths if path in path_ranks
    )
    combined_content = "\n".join(passage.content for passage in passages).casefold()
    forbidden_text_found = tuple(
        text for text in case.forbidden_text if text.casefold() in combined_content
    )
    has_automated_check = bool(
        case.expected_paths or case.forbidden_paths or case.forbidden_text
    )
    passed = (
        not missing_paths and not forbidden_paths_found and not forbidden_text_found
        if has_automated_check
        else None
    )
    serialized_passages = tuple(
        {
            "rank": passage.rank,
            "path": passage.relative_path,
            "page": passage.page,
            "chunk_id": passage.chunk_id,
            "hybrid_score": passage.score,
            "dense_score": passage.metadata.get("dense_score", 0.0),
            "keyword_score": passage.metadata.get("keyword_score", 0.0),
            "content": passage.content,
        }
        for passage in passages
    )
    return EvaluationResult(
        id=case.id,
        category=case.category,
        question=case.question,
        expected_fact=case.expected_fact,
        expected_language=case.expected_language,
        passed=passed,
        expected_path_ranks=expected_path_ranks,
        missing_paths=missing_paths,
        forbidden_paths_found=forbidden_paths_found,
        forbidden_text_found=forbidden_text_found,
        retrieved_paths=tuple(retrieved_paths),
        passages=serialized_passages,
    )


def run_cases(
    cases: Sequence[EvaluationCase],
    vault_root: Path,
    synchronizer: BenchmarkSynchronizer,
    retriever: BenchmarkRetriever,
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for case in cases:
        apply_case_setup(case, vault_root)
        synchronizer.sync_library(raise_on_errors=True)
        results.append(evaluate_retrieval(case, retriever.retrieve(case.question)))
    return results


def summarize_results(results: Sequence[EvaluationResult]) -> dict[str, Any]:
    automated = [result for result in results if result.passed is not None]
    official_recall = [
        result
        for result in results
        if result.category not in {"absent", "conflict"}
        and bool(result.expected_path_ranks or result.missing_paths)
    ]
    official_hits = sum(not result.missing_paths for result in official_recall)
    return {
        "total_cases": len(results),
        "automated_checks": len(automated),
        "automated_passed": sum(result.passed is True for result in automated),
        "automated_failed": sum(result.passed is False for result in automated),
        "manual_only_cases": sum(result.passed is None for result in results),
        "recall_at_5_hits": official_hits,
        "recall_at_5_eligible": len(official_recall),
        "recall_at_5": (
            official_hits / len(official_recall) if official_recall else None
        ),
    }


def _build_and_run(
    cases: Sequence[EvaluationCase], settings: Settings
) -> list[EvaluationResult]:
    ensure_vault(settings)
    embeddings = Qwen3Embeddings(
        settings.embedding_model,
        revision=settings.embedding_revision,
        dimensions=settings.embedding_dimensions,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    manifest = Manifest(settings.manifest_path)
    vector_store = ChromaVectorStore(
        settings.chroma_dir, embeddings, settings.collection_name
    )
    chunker = MultilingualChunker(
        settings.chunk_size,
        settings.chunk_overlap,
        model_name=settings.embedding_model,
        model_revision=settings.embedding_revision,
    )
    synchronizer = Synchronizer(settings, manifest, vector_store, chunker)
    retriever = Retriever(
        vector_store,
        top_k=settings.retrieval_top_k,
        minimum_score=settings.retrieval_min_score,
    )
    sync_result = synchronizer.rebuild_index()
    if not sync_result.successful:
        raise RuntimeError(f"fixture indexing failed: {', '.join(sync_result.errors)}")
    return run_cases(cases, settings.vault_path, synchronizer, retriever)


def _print_results(results: Sequence[EvaluationResult], summary: Mapping[str, Any]) -> None:
    for result in results:
        status = "REVIEW" if result.passed is None else "PASS" if result.passed else "FAIL"
        ranks = ", ".join(
            f"{path}@{rank}" for path, rank in result.expected_path_ranks.items()
        )
        detail = ranks or "no required retrieval path"
        if result.missing_paths:
            detail = f"missing {', '.join(result.missing_paths)}"
        if result.forbidden_paths_found or result.forbidden_text_found:
            detail = "stale evidence retrieved"
        print(f"{status:6} {result.id}  {detail}")
    hits = summary["recall_at_5_hits"]
    eligible = summary["recall_at_5_eligible"]
    percentage = float(summary["recall_at_5"] or 0.0) * 100
    print()
    print(
        f"Automated checks: {summary['automated_passed']}/"
        f"{summary['automated_checks']} passed"
    )
    print(f"Recall@5: {hits}/{eligible} ({percentage:.1f}%)")
    print("Hosted-answer faithfulness, language, citations, and refusals remain manual checks.")


def _parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic 30-case Personal RAG retrieval benchmark."
    )
    parser.add_argument(
        "--evaluation-set",
        type=Path,
        default=project_root / "evaluation" / "evaluation_set.jsonl",
    )
    parser.add_argument(
        "--fixture-library",
        type=Path,
        default=project_root / "evaluation" / "fixture_vault" / "library",
    )
    parser.add_argument("--env-file", type=Path, default=project_root / ".env")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Use a new, non-existent benchmark vault and keep it after the run.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    args = _parser(project_root).parse_args(argv)
    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero")
    cases = load_evaluation_cases(args.evaluation_set.resolve())
    if len(cases) != 30:
        raise SystemExit(f"expected 30 evaluation cases, found {len(cases)}")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="personal-rag-benchmark-", ignore_cleanup_errors=True
        )
        vault_root = Path(temporary.name).resolve()
    else:
        vault_root = args.work_dir.resolve()
        if vault_root.exists():
            raise SystemExit("--work-dir must not already exist")
        vault_root.mkdir(parents=True)

    try:
        shutil.copytree(args.fixture_library.resolve(), vault_root / "library")
        benchmark_environment = dict(os.environ)
        benchmark_environment["KNOWLEDGE_VAULT"] = str(vault_root)
        base_settings = Settings.from_env(
            args.env_file.resolve(), environ=benchmark_environment
        )
        settings = replace(
            base_settings, vault_path=vault_root, retrieval_top_k=args.top_k
        )
        results = _build_and_run(cases, settings)
        summary = summarize_results(results)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = (
            args.output.resolve()
            if args.output
            else project_root / "evaluation" / "results" / f"benchmark-{timestamp}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_set": str(args.evaluation_set.resolve()),
            "fixture_library": str(args.fixture_library.resolve()),
            "configuration": {
                "embedding_model": settings.embedding_model,
                "embedding_revision": settings.embedding_revision,
                "embedding_dimensions": settings.embedding_dimensions,
                "embedding_device": embeddings_device(settings),
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "retrieval_top_k": settings.retrieval_top_k,
                "retrieval_min_score": settings.retrieval_min_score,
                "retrieval": "hybrid_dense_bm25_equal_weight",
            },
            "summary": summary,
            "results": [asdict(result) for result in results],
        }
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _print_results(results, summary)
        print(f"Report: {output_path}")
        return 1 if summary["automated_failed"] else 0
    finally:
        gc.collect()
        if temporary is not None:
            temporary.cleanup()


def embeddings_device(settings: Settings) -> str:
    if settings.embedding_device != "cuda":
        return "cpu_requested"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu_fallback"
    except (ImportError, RuntimeError):
        return "cpu_fallback"


if __name__ == "__main__":
    sys.exit(main())
