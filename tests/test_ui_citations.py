from __future__ import annotations

from personal_rag.ui_citations import source_anchor, source_linked_answer


def test_source_linked_answer_turns_every_known_citation_into_a_superscript_link() -> None:
    rendered = source_linked_answer(
        "First claim [S3], repeated [S3], and another [S12].",
        {"S3", "S12"},
    )

    assert rendered.count('href="#source-s3"') == 2
    assert '<sup><a href="#source-s12"' in rendered
    assert rendered.count('class="source-citation"') == 3
    assert rendered.count("</a></sup>") == 3


def test_source_linked_answer_escapes_model_html_and_leaves_unknown_ids_unlinked() -> None:
    rendered = source_linked_answer(
        "**Bold** <script>alert('x')</script> [S1] [S99]",
        {"S1"},
    )

    assert rendered.startswith("**Bold** &lt;script&gt;")
    assert "<script>" not in rendered
    assert 'href="#source-s1"' in rendered
    assert "[S99]" in rendered
    assert 'href="#source-s99"' not in rendered


def test_source_anchor_targets_the_matching_answer_link() -> None:
    assert source_anchor("S3") == '<span id="source-s3"></span>'
