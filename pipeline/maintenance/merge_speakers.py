from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class SpeakerRecord:
    id: int
    video_id: int
    name: str
    title: str | None
    video_title: str = ""
    youtube_video_id: str = ""


@dataclass(frozen=True)
class DuplicateGroup:
    records: tuple[SpeakerRecord, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self.records)

    @property
    def titles(self) -> tuple[str | None, ...]:
        return tuple(record.title for record in self.records)


@dataclass(frozen=True)
class MergeResult:
    updated_videos: int
    deleted_rows: int


def normalize_text(value: str | None) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(
        "".join(
            character if character.isalnum() else " "
            for character in without_accents.casefold()
        ).split()
    )


def levenshtein_distance(left: str, right: str, max_distance: int = 2) -> int:
    """Calcule la distance, avec abandon rapide au-dela du seuil demande."""
    if left == right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    if len(right) - len(left) > max_distance:
        return max_distance + 1

    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        row_minimum = left_index
        start = max(1, left_index - max_distance)
        end = min(len(right), left_index + max_distance)

        if start > 1:
            current.extend([max_distance + 1] * (start - 1))
        for right_index in range(start, end + 1):
            substitution_cost = int(left_character != right[right_index - 1])
            value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + substitution_cost,
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if end < len(right):
            current.extend([max_distance + 1] * (len(right) - end))
        if row_minimum > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def _preferred_values(values: Iterable[str | None]) -> list[str | None]:
    counts = Counter(values)
    return sorted(
        counts,
        key=lambda value: (
            -counts[value],
            value is None,
            normalize_text(value),
            value or "",
        ),
    )


def _needs_review(records: Sequence[SpeakerRecord]) -> bool:
    raw_names = {record.name for record in records}
    raw_titles = {record.title for record in records}
    return len(raw_names) > 1 or len(raw_titles) > 1


def find_duplicate_groups(
    records: Sequence[SpeakerRecord],
    *,
    max_distance: int = 2,
) -> list[DuplicateGroup]:
    if max_distance not in (0, 1, 2):
        raise ValueError("max_distance doit etre compris entre 0 et 2.")

    records_by_name: dict[str, list[SpeakerRecord]] = defaultdict(list)
    for record in records:
        normalized_name = normalize_text(record.name)
        if normalized_name:
            records_by_name[normalized_name].append(record)

    names = sorted(records_by_name)
    parents = list(range(len(names)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left_name in enumerate(names):
        for right_index in range(left_index + 1, len(names)):
            right_name = names[right_index]
            if abs(len(left_name) - len(right_name)) > max_distance:
                continue
            if levenshtein_distance(left_name, right_name, max_distance) <= max_distance:
                union(left_index, right_index)

    indexes_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(names)):
        indexes_by_root[find(index)].append(index)

    groups: list[DuplicateGroup] = []
    for indexes in indexes_by_root.values():
        grouped_records = sorted(
            (
                record
                for index in indexes
                for record in records_by_name[names[index]]
            ),
            key=lambda record: (normalize_text(record.name), record.video_id, record.id),
        )
        if len(grouped_records) > 1 and _needs_review(grouped_records):
            groups.append(DuplicateGroup(tuple(grouped_records)))

    return sorted(
        groups,
        key=lambda group: (
            normalize_text(_preferred_values(group.names)[0]),
            min(record.id for record in group.records),
        ),
    )


def fetch_speakers(cursor) -> list[SpeakerRecord]:
    cursor.execute(
        """
        SELECT
            speaker.id,
            speaker.video_id,
            speaker.name,
            speaker.title,
            video.title,
            video.youtube_video_id
        FROM speakers speaker
        JOIN videos video ON video.id = speaker.video_id
        ORDER BY speaker.id
        """
    )
    return [
        SpeakerRecord(
            id=int(row[0]),
            video_id=int(row[1]),
            name=str(row[2]).strip(),
            title=(str(row[3]).strip() or None) if row[3] is not None else None,
            video_title=str(row[4] or "").strip(),
            youtube_video_id=str(row[5] or "").strip(),
        )
        for row in cursor.fetchall()
    ]


def merge_group(
    cursor,
    group: DuplicateGroup,
    *,
    canonical_name: str,
    canonical_title: str | None,
) -> MergeResult:
    canonical_name = " ".join(canonical_name.split())
    canonical_title = " ".join((canonical_title or "").split()) or None
    if not canonical_name:
        raise ValueError("Le nom canonique ne peut pas etre vide.")

    records_by_video: dict[int, list[SpeakerRecord]] = defaultdict(list)
    for record in group.records:
        records_by_video[record.video_id].append(record)

    deleted_rows = 0
    for video_records in records_by_video.values():
        survivor = min(
            video_records,
            key=lambda record: (
                record.name != canonical_name,
                record.id,
            ),
        )
        duplicate_ids = [
            record.id for record in video_records if record.id != survivor.id
        ]
        if duplicate_ids:
            cursor.execute(
                "DELETE FROM speakers WHERE id = ANY(%s)",
                (duplicate_ids,),
            )
            deleted_rows += len(duplicate_ids)
        cursor.execute(
            "UPDATE speakers SET name = %s, title = %s WHERE id = %s",
            (canonical_name, canonical_title, survivor.id),
        )

    return MergeResult(
        updated_videos=len(records_by_video),
        deleted_rows=deleted_rows,
    )


def _display_group(
    group: DuplicateGroup,
    index: int,
    total: int,
    output: Callable[[str], None],
) -> None:
    output(f"\n[{index}/{total}] Doublon potentiel ({len(group.records)} occurrence(s))")
    name_counts = Counter(group.names)
    for name in _preferred_values(group.names):
        output(f"  nom: {name} ({name_counts[name]} occurrence(s))")
    title_counts = Counter(group.titles)
    for title in _preferred_values(group.titles):
        label = title or "(aucun poste)"
        output(f"  poste: {label} ({title_counts[title]} occurrence(s))")
    videos = {
        (record.youtube_video_id, record.video_title)
        for record in group.records
    }
    for youtube_id, video_title in sorted(videos):
        output(f"  video: {youtube_id} — {video_title}")


def _choose(
    label: str,
    values: Sequence[str | None],
    *,
    input_func: Callable[[str], str],
    output: Callable[[str], None],
) -> str | None:
    if len(values) == 1:
        return values[0]
    for index, value in enumerate(values, start=1):
        output(f"    {index}. {value or '(aucun poste)'}")
    while True:
        answer = input_func(f"{label} [1]: ").strip()
        if not answer:
            return values[0]
        if answer.lower() in {"s", "q"}:
            return answer.lower()
        if answer.isdigit() and 1 <= int(answer) <= len(values):
            return values[int(answer) - 1]
        output("Choix invalide. Entre un numero, s pour passer ou q pour quitter.")


def review_groups(
    cursor,
    groups: Sequence[DuplicateGroup],
    *,
    dry_run: bool = False,
    input_func: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> tuple[int, int, int]:
    merged_groups = 0
    updated_videos = 0
    deleted_rows = 0

    for index, group in enumerate(groups, start=1):
        _display_group(group, index, len(groups), output)
        if dry_run:
            continue

        names = _preferred_values(group.names)
        chosen_name = _choose(
            "Nom a conserver",
            names,
            input_func=input_func,
            output=output,
        )
        if chosen_name == "q":
            break
        if chosen_name == "s":
            continue

        titles = _preferred_values(group.titles)
        chosen_title = _choose(
            "Poste a conserver",
            titles,
            input_func=input_func,
            output=output,
        )
        if chosen_title == "q":
            break
        if chosen_title == "s":
            continue

        confirmation = input_func(
            f"Fusionner vers {chosen_name!r} / {(chosen_title or '(aucun poste)')!r} ? [o/N] "
        ).strip().casefold()
        if confirmation not in {"o", "oui", "y", "yes"}:
            output("  [skip] groupe ignore")
            continue

        result = merge_group(
            cursor,
            group,
            canonical_name=str(chosen_name),
            canonical_title=chosen_title,
        )
        merged_groups += 1
        updated_videos += result.updated_videos
        deleted_rows += result.deleted_rows
        output(
            f"  [merge] {result.updated_videos} video(s) harmonisee(s), "
            f"{result.deleted_rows} ligne(s) en doublon supprimee(s)"
        )

    return merged_groups, updated_videos, deleted_rows


def run(
    database_url: str,
    *,
    max_distance: int = 2,
    dry_run: bool = False,
    input_func: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> tuple[int, int, int]:
    import psycopg

    with psycopg.connect(
        database_url,
        options="-c search_path=data,public",
    ) as connection:
        with connection.cursor() as cursor:
            records = fetch_speakers(cursor)
            groups = find_duplicate_groups(records, max_distance=max_distance)
            output(
                f"{len(records)} ligne(s) speaker analysee(s), "
                f"{len(groups)} groupe(s) a verifier."
            )
            result = review_groups(
                cursor,
                groups,
                dry_run=dry_run,
                input_func=input_func,
                output=output,
            )
            if dry_run:
                connection.rollback()
                output("Simulation terminee : aucune modification appliquee.")
            else:
                connection.commit()
                output(
                    f"Termine : {result[0]} groupe(s) fusionne(s), "
                    f"{result[1]} video(s) harmonisee(s), "
                    f"{result[2]} ligne(s) supprimee(s)."
                )
            return result
