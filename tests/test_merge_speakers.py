from __future__ import annotations

import unittest

from pipeline.maintenance.merge_speakers import (
    DuplicateGroup,
    SpeakerRecord,
    find_duplicate_groups,
    levenshtein_distance,
    merge_group,
    normalize_text,
    review_groups,
)


def speaker(
    row_id: int,
    video_id: int,
    name: str,
    title: str | None,
) -> SpeakerRecord:
    return SpeakerRecord(
        id=row_id,
        video_id=video_id,
        name=name,
        title=title,
        video_title=f"Video {video_id}",
        youtube_video_id=f"youtube-{video_id}",
    )


class RecordingCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.calls.append((" ".join(sql.split()), params))


class MergeSpeakersTests(unittest.TestCase):
    def test_normalization_ignores_accents_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_text("  Élodie-D'ARC  "), "elodie d arc")

    def test_levenshtein_accepts_one_or_two_changed_letters(self) -> None:
        self.assertEqual(levenshtein_distance("alice martin", "alise martin"), 1)
        self.assertEqual(levenshtein_distance("alice martin", "alixe marton"), 2)
        self.assertGreater(levenshtein_distance("alice martin", "bob durand"), 2)

    def test_finds_exact_title_conflicts_and_close_name_variants(self) -> None:
        records = [
            speaker(1, 10, "Alice Martin", "Directrice"),
            speaker(2, 11, "Alice Martin", "Fondatrice"),
            speaker(3, 12, "Alise Martin", "Directrice"),
            speaker(4, 13, "Bob Durand", "Journaliste"),
            speaker(5, 14, "Bob Durand", "Journaliste"),
        ]

        groups = find_duplicate_groups(records, max_distance=2)

        self.assertEqual(len(groups), 1)
        self.assertEqual({record.id for record in groups[0].records}, {1, 2, 3})

    def test_distance_option_can_require_exact_normalized_names(self) -> None:
        records = [
            speaker(1, 10, "Élodie Martin", "Directrice"),
            speaker(2, 11, "elodie martin", "Fondatrice"),
            speaker(3, 12, "Elodia Martin", "Directrice"),
        ]

        groups = find_duplicate_groups(records, max_distance=0)

        self.assertEqual(len(groups), 1)
        self.assertEqual({record.id for record in groups[0].records}, {1, 2})

    def test_merge_keeps_one_row_per_video_and_harmonizes_every_occurrence(self) -> None:
        group = DuplicateGroup(
            (
                speaker(1, 10, "Alise Martin", "CEO"),
                speaker(2, 10, "Alice Martin", "Fondatrice"),
                speaker(3, 11, "Alise Martin", None),
            )
        )
        cursor = RecordingCursor()

        result = merge_group(
            cursor,
            group,
            canonical_name="Alice Martin",
            canonical_title="Fondatrice",
        )

        self.assertEqual(result.updated_videos, 2)
        self.assertEqual(result.deleted_rows, 1)
        self.assertEqual(
            cursor.calls,
            [
                ("DELETE FROM speakers WHERE id = ANY(%s)", ([1],)),
                (
                    "UPDATE speakers SET name = %s, title = %s WHERE id = %s",
                    ("Alice Martin", "Fondatrice", 2),
                ),
                (
                    "UPDATE speakers SET name = %s, title = %s WHERE id = %s",
                    ("Alice Martin", "Fondatrice", 3),
                ),
            ],
        )

    def test_interactive_review_chooses_name_and_title_before_merging(self) -> None:
        group = DuplicateGroup(
            (
                speaker(1, 10, "Alice Martin", "Directrice"),
                speaker(2, 11, "Alise Martin", "Fondatrice"),
            )
        )
        answers = iter(["1", "2", "oui"])
        output: list[str] = []
        cursor = RecordingCursor()

        result = review_groups(
            cursor,
            [group],
            input_func=lambda _prompt: next(answers),
            output=output.append,
        )

        self.assertEqual(result, (1, 2, 0))
        updates = [call for call in cursor.calls if call[0].startswith("UPDATE")]
        self.assertEqual(len(updates), 2)
        self.assertTrue(all(call[1][0] == "Alice Martin" for call in updates))
        self.assertTrue(all(call[1][1] == "Fondatrice" for call in updates))

    def test_dry_run_only_displays_proposals(self) -> None:
        group = DuplicateGroup(
            (
                speaker(1, 10, "Alice Martin", "Directrice"),
                speaker(2, 11, "Alise Martin", "Fondatrice"),
            )
        )
        cursor = RecordingCursor()

        result = review_groups(cursor, [group], dry_run=True, output=lambda _line: None)

        self.assertEqual(result, (0, 0, 0))
        self.assertEqual(cursor.calls, [])


if __name__ == "__main__":
    unittest.main()
