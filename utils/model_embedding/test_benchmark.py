import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.model_embedding.benchmark import (
    Chunk,
    EmbeddingConfig,
    EvalCase,
    chunk_key,
    content_hash,
    discover_chunks,
    evaluate_configuration,
    load_cases,
    write_reports,
)
from utils.model_embedding.generate_cases import (
    CONTENU_COURANT,
    PRENOM_COURT,
    build_cases,
    parse_generated_questions,
    select_chunks,
    select_generation_groups,
)


class ModelEmbeddingBenchmarkTests(unittest.TestCase):
    def test_chunk_key_normalizes_integer_float(self):
        self.assertEqual(chunk_key("video-a", 7.0), "video-a:7")

    def test_load_cases_accepts_multiple_relevant_chunks(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "cases.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "question": "Question ?",
                        "relevant": [
                            {"video_key": "video-a", "chunk_index": 1},
                            "video-a:2",
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cases = load_cases(path)
        self.assertEqual(cases[0].relevant, ("video-a:1", "video-a:2"))

    def test_discover_chunks_prefers_current_transcript_chunks_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            video_dir = Path(temporary_dir) / "video-a"
            chunks_dir = video_dir / "outputs" / "chunks"
            chunks_dir.mkdir(parents=True)
            (chunks_dir / "transcript_chunks.json").write_text(
                json.dumps({"chunks": [{"chunk_index": 1, "content": "ancien"}]}),
                encoding="utf-8",
            )
            (chunks_dir / "transcript_chunks_speaker_validated.json").write_text(
                json.dumps({"chunks": [{"chunk_index": 1, "content": "valide"}]}),
                encoding="utf-8",
            )
            chunks = discover_chunks(Path(temporary_dir))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].key, "video-a:1")
        self.assertEqual(chunks[0].content, "ancien")

    def test_evaluate_configuration_computes_recall_and_mrr(self):
        chunks = [
            Chunk("v:1", "v", "1", "alpha", "source"),
            Chunk("v:2", "v", "2", "beta", "source"),
            Chunk("v:3", "v", "3", "gamma", "source"),
        ]
        cases = [
            EvalCase("q1", "alpha", ("v:1",)),
            EvalCase("q2", "beta", ("v:2",)),
        ]
        vectors = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        questions = np.asarray([[1, 0], [0.6, 0.8]], dtype=np.float32)
        summary, details = evaluate_configuration(
            EmbeddingConfig("test", "test-model", 2),
            chunks,
            cases,
            vectors,
            questions,
            [1, 2],
        )
        self.assertEqual(summary["recall_at_1"], 1.0)
        self.assertEqual(summary["recall_at_2"], 1.0)
        self.assertEqual(summary["mrr"], 1.0)
        self.assertEqual(details[0]["first_relevant_rank"], 1)

    def test_content_hash_is_order_independent(self):
        first = content_hash([("b", "deux"), ("a", "un")])
        second = content_hash([("a", "un"), ("b", "deux")])
        self.assertEqual(first, second)

    def test_select_chunks_balances_videos(self):
        chunks = [
            Chunk(f"a:{index}", "a", str(index), "a" * 200, "source")
            for index in range(1, 5)
        ] + [
            Chunk(f"b:{index}", "b", str(index), "b" * 200, "source")
            for index in range(1, 5)
        ]
        selected = select_chunks(chunks, count=4, seed=42, min_chars=100, max_per_video=0)
        self.assertEqual({chunk.video_key for chunk in selected}, {"a", "b"})
        self.assertEqual(sum(chunk.video_key == "a" for chunk in selected), 2)
        self.assertEqual(sum(chunk.video_key == "b" for chunk in selected), 2)

    def test_parse_generated_questions_validates_all_source_keys(self):
        payload = json.dumps(
            [
                {"source_key": "v:1", "question": "Quelle information est donnée ici ?"},
                {"source_key": "v:2", "question": "Quel débouché est explicitement mentionné ?"},
            ],
            ensure_ascii=False,
        )
        chunks = [
            Chunk("v:1", "v", "1", "contenu", "source"),
            Chunk("v:2", "v", "2", "contenu", "source"),
        ]
        questions = parse_generated_questions(payload, chunks, CONTENU_COURANT)
        self.assertEqual(set(questions), {"v:1", "v:2"})

    def test_parse_generated_questions_rejects_ambiguous_reference(self):
        payload = json.dumps(
            [{"source_key": "v:1", "question": "Pourquoi cette personne a-t-elle choisi cette école ?"}],
            ensure_ascii=False,
        )
        with self.assertRaisesRegex(ValueError, "Question non autonome"):
            parse_generated_questions(
                payload,
                [Chunk("v:1", "v", "1", "contenu", "source")],
                CONTENU_COURANT,
            )

    def test_content_question_rejects_vague_program_reference(self):
        payload = json.dumps(
            [{"source_key": "v:1", "question": "Pourquoi cette formation a-t-elle été choisie ?"}],
            ensure_ascii=False,
        )
        with self.assertRaisesRegex(ValueError, "contenu invalide"):
            parse_generated_questions(
                payload,
                [Chunk("v:1", "v", "1", "contenu", "source")],
                CONTENU_COURANT,
            )

    def test_build_cases_uses_selected_chunk_as_relevant(self):
        selected = [Chunk("video-a:7", "video-a", "7", "contenu", "source")]
        cases = build_cases(
            selected,
            {"video-a:7": "Quelle réponse est présente ?"},
            {"video-a:7": CONTENU_COURANT},
        )
        self.assertEqual(cases[0]["id"], "q001")
        self.assertEqual(cases[0]["category"], CONTENU_COURANT)
        self.assertEqual(cases[0]["relevant"], [{"video_key": "video-a", "chunk_index": 7}])

    def test_person_question_requires_first_name_and_rejects_last_name(self):
        chunk = Chunk("v:1", "v", "1", "contenu", "source", ("Alice Martin",))
        valid = json.dumps(
            [{"source_key": "v:1", "question": "Quel métier Alice souhaite-t-elle exercer ?"}],
            ensure_ascii=False,
        )
        self.assertIn("v:1", parse_generated_questions(valid, [chunk], PRENOM_COURT))
        invalid = json.dumps(
            [{"source_key": "v:1", "question": "Quel métier Alice Martin souhaite-t-elle exercer ?"}],
            ensure_ascii=False,
        )
        with self.assertRaisesRegex(ValueError, "nom de famille"):
            parse_generated_questions(invalid, [chunk], PRENOM_COURT)

    def test_select_generation_groups_assigns_two_categories(self):
        chunks = []
        for video_index in range(1, 7):
            chunks.append(
                Chunk(
                    f"v{video_index}:1",
                    f"v{video_index}",
                    "1",
                    "x" * 200,
                    "source",
                    (f"Prenom{video_index} Nom{video_index}",),
                )
            )
        selected, categories = select_generation_groups(
            chunks,
            person_count=3,
            content_count=3,
            seed=42,
            min_chars=100,
            max_per_video=0,
        )
        self.assertEqual(len(selected), 6)
        self.assertEqual(list(categories.values()).count(PRENOM_COURT), 3)
        self.assertEqual(list(categories.values()).count(CONTENU_COURANT), 3)

    def test_write_reports_creates_readable_markdown_files(self):
        summaries = [
            {
                "configuration": "large-1024",
                "model": "text-embedding-3-large",
                "dimensions": 1024,
                "evaluated_questions": 1,
                "recall_at_1": 1.0,
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "bytes_per_vector": 4104,
                "chunk_prompt_tokens": 10,
                "question_prompt_tokens": 2,
                "embedding_seconds": 0.5,
            }
        ]
        details = [
            {
                "configuration": "large-1024",
                "case_id": "q001",
                "question": "Quel poste occupe Alice ?",
                "category": PRENOM_COURT,
                "answerable": True,
                "relevant": ["video-a:1"],
                "first_relevant_rank": 1,
                "top_results": [{"key": "video-a:1", "score": 0.9}],
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            reports_dir = Path(temporary_dir)
            write_reports(reports_dir, summaries, details, [1, 5])
            report = (reports_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("100.0 %", report)
            self.assertIn("## 1. Résumé global", report)
            self.assertIn("## 2. Résultats par catégorie", report)
            self.assertIn("## 3. Définition des métriques", report)
            self.assertIn("## 4. Échecs au Recall@5", report)
            self.assertIn("## 5. Détail par question", report)
            self.assertIn("Mean Reciprocal Rank", report)
            self.assertIn(PRENOM_COURT, report)
            self.assertIn("Score cosinus", report)
            self.assertIn("video-a:1", report)
            self.assertEqual(
                sorted(path.name for path in reports_dir.iterdir()),
                ["report.md"],
            )


if __name__ == "__main__":
    unittest.main()
