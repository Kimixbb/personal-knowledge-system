from __future__ import annotations

import json
from pathlib import Path

from personal_rag.evaluation import (
    EvaluationCase,
    apply_case_setup,
    evaluate_retrieval,
    load_evaluation_cases,
)
from personal_rag.retrieval import RetrievedPassage


def passage(rank: int, path: str, content: str) -> RetrievedPassage:
    return RetrievedPassage(
        source_id=f"S{rank}",
        rank=rank,
        score=1.0 / rank,
        chunk_id=f"chunk-{rank}",
        content=content,
        relative_path=path,
        page=None,
        metadata={"dense_score": 0.5, "keyword_score": 0.5},
    )


def test_repository_evaluation_set_has_thirty_ordered_cases() -> None:
    evaluation_path = Path(__file__).parents[1] / "evaluation" / "evaluation_set.jsonl"

    cases = load_evaluation_cases(evaluation_path)

    assert [case.id for case in cases] == [f"q{index:02d}" for index in range(1, 31)]
    assert cases[28].setup is not None
    assert cases[29].setup is not None


def test_load_evaluation_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    case = {
        "id": "q01",
        "category": "example",
        "question": "Question?",
        "expected_paths": [],
        "expected_fact": "fact",
        "expected_language": "en",
    }
    path.write_text(
        "\n".join((json.dumps(case), json.dumps(case))), encoding="utf-8"
    )

    try:
        load_evaluation_cases(path)
    except ValueError as exc:
        assert "duplicate evaluation case id" in str(exc)
    else:
        raise AssertionError("duplicate IDs must be rejected")


def test_evaluate_retrieval_reports_required_path_ranks_and_missing_paths() -> None:
    case = EvaluationCase(
        id="q21",
        category="multiple_facts",
        question="Summarize Project Cedar.",
        expected_paths=("library/english-notes.md", "library/current-plan.txt"),
        expected_fact="Daniel Cho; October 12, 2026; USD 18,400",
        expected_language="en",
    )

    result = evaluate_retrieval(
        case,
        [
            passage(1, "library/english-notes.md", "Daniel Cho owns Cedar."),
            passage(2, "library/unrelated.md", "Unrelated."),
        ],
    )

    assert result.passed is False
    assert result.expected_path_ranks == {"library/english-notes.md": 1}
    assert result.missing_paths == ("library/current-plan.txt",)


def test_absent_case_is_executed_but_not_automatically_scored() -> None:
    case = EvaluationCase(
        id="q25",
        category="absent",
        question="What is Mei Lin's phone number?",
        expected_paths=(),
        expected_fact="insufficient evidence",
        expected_language="en",
    )

    result = evaluate_retrieval(
        case, [passage(1, "library/english-notes.md", "Mei Lin likes cake.")]
    )

    assert result.passed is None
    assert result.requires_manual_answer_review is True
    assert result.retrieved_paths == ("library/english-notes.md",)


def test_stale_text_and_deleted_path_fail_automated_checks() -> None:
    case = EvaluationCase(
        id="q30",
        category="deleted_document",
        question="Is the old date still retrieved?",
        expected_paths=("library/current-plan.txt",),
        expected_fact="No stale date",
        expected_language="en",
        forbidden_paths=("library/old-plan.txt",),
        forbidden_text=("October 5, 2026",),
    )

    result = evaluate_retrieval(
        case,
        [
            passage(1, "library/current-plan.txt", "October 12, 2026"),
            passage(2, "library/old-plan.txt", "October 5, 2026"),
        ],
    )

    assert result.passed is False
    assert result.forbidden_paths_found == ("library/old-plan.txt",)
    assert result.forbidden_text_found == ("October 5, 2026",)


def test_case_setup_replaces_text_and_deletes_only_inside_vault(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    notes = library / "notes.md"
    notes.write_text("Birthday: August 7", encoding="utf-8")
    obsolete = library / "obsolete.txt"
    obsolete.write_text("old", encoding="utf-8")

    replace_case = EvaluationCase(
        id="q29",
        category="edited_document",
        question="New birthday?",
        expected_paths=("library/notes.md",),
        expected_fact="August 9",
        expected_language="en",
        setup={
            "action": "replace_text",
            "path": "library/notes.md",
            "old": "August 7",
            "new": "August 9",
        },
    )
    delete_case = EvaluationCase(
        id="q30",
        category="deleted_document",
        question="Still present?",
        expected_paths=(),
        expected_fact="No",
        expected_language="en",
        setup={"action": "delete", "path": "library/obsolete.txt"},
    )

    apply_case_setup(replace_case, tmp_path)
    apply_case_setup(delete_case, tmp_path)

    assert notes.read_text(encoding="utf-8") == "Birthday: August 9"
    assert not obsolete.exists()
