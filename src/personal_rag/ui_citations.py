from __future__ import annotations

import html
import re


_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def _source_target_id(source_id: str) -> str:
    if re.fullmatch(r"S\d+", source_id) is None:
        raise ValueError(f"Invalid source ID: {source_id}")
    return f"source-{source_id.lower()}"


def source_linked_answer(answer: str, source_ids: set[str]) -> str:
    """Render known source IDs as safe links to their source cards."""
    escaped_answer = html.escape(answer)

    def replace_citation(match: re.Match[str]) -> str:
        source_id = match.group(1)
        if source_id not in source_ids:
            return match.group(0)
        target_id = _source_target_id(source_id)
        return (
            f'<sup><a href="#{target_id}" class="source-citation" '
            f'aria-label="Jump to source {source_id}">[{source_id}]</a></sup>'
        )

    return _CITATION_PATTERN.sub(replace_citation, escaped_answer)


def source_anchor(source_id: str) -> str:
    """Return the trusted HTML anchor targeted by an answer citation."""
    return f'<span id="{_source_target_id(source_id)}"></span>'
