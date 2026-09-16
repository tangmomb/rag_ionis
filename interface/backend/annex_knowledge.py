from __future__ import annotations

import re
import unicodedata
from typing import Any

from interface.backend.config import (
    PERSON_SCOPED_KNOWLEDGE_FILE,
    QUESTION_SCOPED_KNOWLEDGE_FILE,
)


def _normalise_person_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    without_punctuation = re.sub(r"[^\w\s]", " ", without_accents)
    return re.sub(r"\s+", " ", without_punctuation).strip().casefold()


def _parse_annex(raw_text: str) -> tuple[list[str], str]:
    """Read the two explicit sections of the plain-text annex format."""
    sections: dict[str, list[str]] = {"persons": [], "knowledge": []}
    active_section: str | None = None
    markers = {
        "[PERSONNES]": "persons",
        "[/PERSONNES]": None,
        "[CONNAISSANCES]": "knowledge",
        "[/CONNAISSANCES]": None,
    }
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if line in markers:
            active_section = markers[line]
            continue
        # Les commentaires sont écrits avec un seul `# ` ; les titres Markdown
        # (`## Nom`) restent bien dans les connaissances envoyées au modèle.
        if not active_section or not line or line == "#" or line.startswith("# "):
            continue
        sections[active_section].append(raw_line)

    persons = [line.strip() for line in sections["persons"]]
    knowledge = "\n".join(sections["knowledge"]).strip()
    return persons, knowledge


def matching_person_scoped_annex_persons(persons: list[str]) -> list[str]:
    """Return planner-extracted people that are explicitly allow-listed locally."""
    requested_persons = [str(person).strip() for person in persons if str(person).strip()]
    if not requested_persons or not PERSON_SCOPED_KNOWLEDGE_FILE.is_file():
        return []
    allowed_persons, _ = _parse_annex(
        PERSON_SCOPED_KNOWLEDGE_FILE.read_text(encoding="utf-8")
    )
    allowed_names = {_normalise_person_name(person) for person in allowed_persons}
    return [
        person
        for person in requested_persons
        if _normalise_person_name(person) in allowed_names
    ]


def load_person_scoped_knowledge(persons: list[str]) -> tuple[str | None, dict[str, Any]]:
    """Load the annex only for people both resolved and explicitly allow-listed."""
    resolved_persons = [str(person).strip() for person in persons if str(person).strip()]
    trace: dict[str, Any] = {
        "enabled": False,
        "persons": resolved_persons,
        "path": PERSON_SCOPED_KNOWLEDGE_FILE.name,
    }
    if not resolved_persons:
        trace["reason"] = "no_execution_plan_persons"
        return None, trace
    if not PERSON_SCOPED_KNOWLEDGE_FILE.is_file():
        trace["reason"] = "file_not_found"
        return None, trace

    _, knowledge = _parse_annex(
        PERSON_SCOPED_KNOWLEDGE_FILE.read_text(encoding="utf-8")
    )
    matched_persons = matching_person_scoped_annex_persons(resolved_persons)
    trace["matched_persons"] = matched_persons
    if not matched_persons:
        trace["reason"] = "no_matching_allowed_person"
        return None, trace
    if not knowledge:
        trace["reason"] = "empty_knowledge_section"
        return None, trace

    trace.update(
        {
            "enabled": True,
            "reason": "loaded",
            "character_count": len(knowledge),
        }
    )
    return knowledge, trace


def _parse_question_annex(raw_text: str) -> tuple[list[str], str]:
    """Read exact question triggers and knowledge from the generic annex."""
    sections: dict[str, list[str]] = {"triggers": [], "knowledge": []}
    active_section: str | None = None
    markers = {
        "[DECLENCHEURS]": "triggers",
        "[/DECLENCHEURS]": None,
        "[CONNAISSANCES]": "knowledge",
        "[/CONNAISSANCES]": None,
    }
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if line in markers:
            active_section = markers[line]
            continue
        # Conserver les titres Markdown des connaissances, ignorer seulement
        # les commentaires de configuration écrits avec `# `.
        if not active_section or not line or line == "#" or line.startswith("# "):
            continue
        sections[active_section].append(raw_line)
    return (
        [line.strip() for line in sections["triggers"]],
        "\n".join(sections["knowledge"]).strip(),
    )


def load_question_scoped_knowledge(question: str) -> tuple[str | None, dict[str, Any]]:
    """Load generic annex knowledge only for a configured complete question."""
    trace: dict[str, Any] = {
        "enabled": False,
        "path": QUESTION_SCOPED_KNOWLEDGE_FILE.name,
    }
    if not QUESTION_SCOPED_KNOWLEDGE_FILE.is_file():
        trace["reason"] = "file_not_found"
        return None, trace

    triggers, knowledge = _parse_question_annex(
        QUESTION_SCOPED_KNOWLEDGE_FILE.read_text(encoding="utf-8")
    )
    normalised_question = _normalise_person_name(question)
    matched_triggers = [
        trigger
        for trigger in triggers
        if (normalised_trigger := _normalise_person_name(trigger))
        and normalised_trigger == normalised_question
    ]
    trace["matched_triggers"] = matched_triggers
    if not matched_triggers:
        trace["reason"] = "no_matching_trigger"
        return None, trace
    if not knowledge:
        trace["reason"] = "empty_knowledge_section"
        return None, trace

    trace.update(
        {
            "enabled": True,
            "reason": "loaded",
            "character_count": len(knowledge),
        }
    )
    return knowledge, trace
