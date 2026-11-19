from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from utils.model_embedding.generate_synthetic_questions import (
    batch_request, daily_scenarios, validated_question,
)
from utils.model_embedding.simulate_rag_users import followup_request, play_turns, simulate
from utils.model_embedding import schedule_synthetic_users as schedule
from utils.model_embedding.schedule_synthetic_users import (
    API_URL, PARIS, build_plan, dispatch_followups, open_sessions, submit_followups,
)


class SyntheticConversationTests(unittest.TestCase):
    def test_new_scheduled_conversations_have_only_one_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            state = open_sessions(Path(directory) / "simulation.json", [])
        self.assertEqual(state["max_turns"], 2)

    def test_public_dispatch_stops_at_the_configured_daily_quota(self):
        sessions = [
            {"id": str(index), "next_question": f"Question {index} ?", "turns": []}
            for index in range(3)
        ]
        state = {"run_id": "quota", "request_limit": 2, "sessions": sessions}
        with tempfile.TemporaryDirectory() as directory, patch(
            "utils.model_embedding.simulate_rag_users.ask_rag",
            return_value={"status": "completed", "response": {"conversation_id": 10}},
        ) as ask:
            manifest = Path(directory) / "simulation.json"
            play_turns(state, manifest, sessions, 1, 2, "https://example.test/api/rag/stream")
            self.assertEqual(ask.call_count, 2)
            self.assertEqual(sessions[2]["outcome"], "request_quota_reached")
            resumed = json.loads(manifest.read_text(encoding="utf-8"))
            play_turns(resumed, manifest, resumed["sessions"][2:], 1, 2, "https://example.test/api/rag/stream")
            self.assertEqual(ask.call_count, 2)

    def test_followup_uses_last_exchange_and_uncited_generation_chunks(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = {"data": [{"attributes": {"input.value": json.dumps({
            "sources": [
                {"text": "Extrait cité.", "global_context": {"text": "Résumé global."}},
                {"text": "Extrait non cité.", "global_context": {"text": "Résumé global."}},
            ],
        })}}]}
        session = {
            "id": "test",
            "turns": [
                {"question": "Question précédente ?", "response": {"answer": "Ancienne réponse."}},
                {
                    "question": "Question du dernier échange ?",
                    "response": {
                        "answer": "Dernière réponse.",
                        "retrieval": {"telemetry": {"project": "rag-ionis-synthetic", "trace_id": "trace"}},
                        "sources": [{"text": "Extrait cité."}],
                    },
                },
            ],
        }
        with patch("utils.model_embedding.simulate_rag_users.requests.get", return_value=response) as get:
            request = followup_request(session, 4096)

        self.assertEqual(get.call_args.kwargs["params"], {"name": "generation", "trace_id": "trace"})
        self.assertEqual(json.loads(request["body"]["messages"][1]["content"]), {
            "question_initiale": "Question du dernier échange ?",
            "chunks_envoyes_par_le_systeme": ["Extrait cité.", "Résumé global.", "Extrait non cité."],
            "reponse_initiale": "Dernière réponse.",
        })

    def test_opening_uses_excerpts_and_returns_only_the_question(self):
        scenario = {
            "id": "test", "video": "Un témoignage",
            "chunks": ["Le premier passage.", "Le deuxième passage."],
        }

        def response(content):
            return {"choices": [{"message": {"content": content}}]}

        request = batch_request(scenario)["body"]
        self.assertEqual(request["response_format"], {"type": "text"})
        self.assertEqual(json.loads(request["messages"][1]["content"]), {
            "video": scenario["video"], "chunks": scenario["chunks"],
        })
        question = "Quel outil conseille-t-elle ?"
        generated = validated_question(response(question), scenario)
        self.assertEqual(generated, {"id": "test", "question": question})
        self.assertEqual(validated_question(response(json.dumps({"question": question})), scenario), generated)
        with self.assertRaises(ValueError):
            validated_question(response(f"{question}\nPourquoi ?"), scenario)

    def test_video_selection_keeps_all_short_chunks_and_one_long_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            def write_video(name, duration, rows):
                base = Path(directory) / name
                metadata = base / "metadata" / "youtube_video_metadata.json"
                metadata.parent.mkdir(parents=True)
                metadata.write_text(json.dumps({"title": name, "duration_seconds": duration}), encoding="utf-8")
                chunks = base / "outputs" / "chunks" / "transcript_chunks.json"
                chunks.parent.mkdir(parents=True)
                chunks.write_text(json.dumps({"chunks": rows}), encoding="utf-8")

            details = [
                {"chunk_level": "detail", "content": "Premier passage."},
                {"chunk_level": "detail", "content": "Deuxième passage."},
            ]
            write_video("Courte", 600, details)
            write_video("Longue", 601, [*details, {"chunk_level": "global", "content": "Résumé global."}])
            day = date(2026, 11, 19)
            selection = daily_scenarios(Path(directory), day, 2)
            self.assertEqual(selection, daily_scenarios(Path(directory), day, 2))

        self.assertEqual({row["video"]: row["chunks"] for row in selection}, {
            "Courte": ["Premier passage.", "Deuxième passage."],
            "Longue": ["Résumé global."],
        })

    def test_daily_schedule_uses_unique_minutes_in_the_morning(self):
        day = date(2026, 11, 19)
        questions = [{"id": str(index)} for index in range(30)]
        plan = build_plan(questions, day)
        self.assertEqual(plan, build_plan(questions, day))
        self.assertEqual([len(group) for group in plan], [30])
        self.assertEqual(
            {row["id"] for group in plan for row in group}, {row["id"] for row in questions}
        )
        times = [datetime.fromisoformat(row["send_at"]) for group in plan for row in group]
        self.assertEqual(times, sorted(set(times)))
        self.assertTrue(all(item.date() == day and 8 <= item.astimezone(PARIS).hour < 12 for item in times))
        tomorrow = build_plan(questions, day + timedelta(days=1))
        self.assertNotEqual(
            [time.time() for time in times],
            [datetime.fromisoformat(row["send_at"]).time() for group in tomorrow for row in group],
        )

    def test_scheduled_phases_use_the_four_paris_times(self):
        units = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
        expected = {
            "night": ("02:15", "night"),
            "day": ("08:00", "openings"),
            "followup-batch": ("12:30", "followup-batch"),
            "followup-send": ("16:00", "followup-send"),
        }
        self.assertEqual(
            {path.name for path in units.glob("rag-ionis-synthetic-*.timer")},
            {f"rag-ionis-synthetic-{name}.timer" for name in expected},
        )
        for name, (time, phase) in expected.items():
            with self.subTest(phase=phase):
                text = (units / f"rag-ionis-synthetic-{name}.timer").read_text(encoding="utf-8")
                self.assertIn(f"OnCalendar=*-*-* {time}:00 Europe/Paris", text)
                self.assertIn(f"Unit=rag-ionis-synthetic@{phase}.service", text)

    def test_persistent_followup_timer_cannot_start_before_todays_window(self):
        with (
            patch("sys.argv", ["schedule", "followup-batch"]),
            patch.object(schedule, "datetime", wraps=datetime) as clock,
            patch.object(schedule, "dotenv_values") as load_env,
        ):
            clock.now.return_value = datetime(2026, 11, 19, 8, tzinfo=PARIS)
            self.assertEqual(schedule.main(), 0)
        load_env.assert_not_called()

    def test_batch_service_persists_the_job_and_dispatch_resumes_it_once(self):
        day = datetime.now(PARIS).date() + timedelta(days=1)
        client = MagicMock()
        running = SimpleNamespace(id="stored-job", status="RUNNING")
        client.batch.jobs.create.return_value = running
        client.batch.jobs.get.side_effect = [running, SimpleNamespace(
            id="stored-job", status="SUCCESS", output_file=None, outputs=[{
                "custom_id": "1", "response": {"body": {"choices": [{
                    "message": {"content": '{"question": "Peux-tu détailler ?"}'},
                }]}},
            }],
        )]
        state = {
            "api_url": API_URL, "max_turns": 2, "run_id": "test",
            "sessions": [
                {"id": "1", "turns": [{
                    "status": "completed", "question": "Question initiale ?",
                    "response": {"answer": "Exemple", "conversation_id": 42},
                }]},
                {"id": "2", "turns": [], "next_question": "Question manquée ?"},
            ],
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("utils.model_embedding.generate_synthetic_questions.time.sleep"),
            patch("utils.model_embedding.simulate_rag_users.wait_until") as wait,
            patch("utils.model_embedding.simulate_rag_users.ask_rag", return_value={
                "status": "completed", "response": {"answer": "Plus de détails", "conversation_id": 42},
            }) as ask,
        ):
            root = Path(directory)
            (root / "day-plan.json").write_text('[[{"id": "1"}, {"id": "2"}]]', encoding="utf-8")
            manifest = root / "vague-1" / "simulation.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps(state), encoding="utf-8")

            self.assertEqual(submit_followups(client, day, root), 0)
            self.assertEqual(submit_followups(client, day, root), 0)
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["turn_2_job_1"], "stored-job")
            client.batch.jobs.get.assert_not_called()
            ask.assert_not_called()

            self.assertEqual(dispatch_followups(client, day, root), 0)
            self.assertEqual(dispatch_followups(client, day, root), 0)
            saved = json.loads(manifest.read_text(encoding="utf-8"))

        client.batch.jobs.create.assert_called_once()
        self.assertEqual(client.batch.jobs.get.call_args_list, [call(job_id="stored-job")] * 2)
        ask.assert_called_once()
        self.assertEqual(ask.call_args.args[1]["question"], "Peux-tu détailler ?")
        self.assertEqual(ask.call_args.args[2], 42)
        wait.assert_called_once_with(datetime(day.year, day.month, day.day, 16, tzinfo=PARIS))
        self.assertEqual([session["outcome"] for session in saved["sessions"]], ["turn_limit", "missed_opening"])

    def test_followup_ready_after_cutoff_is_never_sent(self):
        day = date(2026, 11, 19)
        session = {"turns": [{"status": "completed"}], "next_question": "Et ensuite ?"}
        state = {"sessions": [session], "max_turns": 4}
        client = MagicMock()
        deadline = datetime(2026, 11, 19, 20, tzinfo=PARIS)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("utils.model_embedding.simulate_rag_users.ask_rag") as ask,
            patch("utils.model_embedding.simulate_rag_users.datetime") as clock,
        ):
            clock.now.return_value = deadline
            self.assertEqual(simulate(
                client, state, Path(directory) / "simulation.json", day, 1, 5,
                send_deadline=deadline,
            ), 0)
            ask.assert_not_called()
        self.assertEqual(session["outcome"], "followup_window_closed")

    def test_batches_followups_after_streams_and_reuses_conversation(self):
        requests = []
        ended = set()
        waits = []
        send_time = datetime(2026, 11, 19, 16, tzinfo=PARIS)
        state = {
            "api_url": "http://localhost/api/rag/stream",
            "run_id": "test",
            "max_turns": 2,
            "sessions": [
                {
                    "id": str(index), "profile": "étudiant",
                    "expected_answer": "référence privée du test",
                    "next_question": f"Question {index} ?", "turns": [],
                }
                for index in (1, 2)
            ],
        }

        def post(_url, *, json, headers, **_kwargs):
            question = json["question"]
            self.assertEqual(waits, [send_time] if json["conversationId"] is not None else [])
            conversation_id = json["conversationId"] or int(question.split()[1])
            requests.append((question, json["conversationId"], headers["X-Request-ID"]))
            self.assertNotIn("expected_answer", json)
            answer = {"answer": f"Exemple {conversation_id}", "conversation_id": conversation_id}
            response = MagicMock(status_code=200, headers={"Content-Type": "text/event-stream"})
            response.__enter__.return_value = response

            def lines(*_args, **_kwargs):
                for event in (
                    {"type": "chunk", "text": "Exemple"},
                    {"type": "done", "response": answer},
                    {"type": "memory", "status": "completed"},
                ):
                    yield b"data: " + encode(event).encode()
                    yield b""
                ended.add(question)

            response.iter_lines.side_effect = lines
            return response

        def encode(value):
            return json.dumps(value)

        def result(custom_id, question):
            return {
                "custom_id": custom_id,
                "response": {"body": {"choices": [{
                    "message": {"content": [{"type": "text", "text": encode({"question": question})}]},
                }]}},
            }

        def create(*, requests, model, **_kwargs):
            self.assertEqual(model, "mistral-large-4")
            self.assertEqual(ended, {"Question 1 ?", "Question 2 ?"})
            self.assertEqual(waits, [])
            ids = [row["custom_id"] for row in requests]
            first_context = json.loads(requests[0]["body"]["messages"][1]["content"])
            self.assertEqual(first_context, {
                "question_initiale": "Question 1 ?",
                "chunks_envoyes_par_le_systeme": [],
                "reponse_initiale": "Exemple 1",
            })
            self.assertNotIn("référence privée", str(first_context))
            if ids == ["1", "2"]:
                outputs = [result("2", None), result("1", "Question 1 ?")]
            else:
                self.assertEqual(ids, ["1"])
                outputs = [result("1", "Peux-tu détailler cet exemple ?")]
            return SimpleNamespace(id=f"job-{len(ids)}", status="SUCCESS", output_file=None, outputs=outputs)

        client = MagicMock()
        client.batch.jobs.create.side_effect = create
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("utils.model_embedding.simulate_rag_users.requests.post", side_effect=post),
            patch("utils.model_embedding.simulate_rag_users.wait_until", side_effect=waits.append),
        ):
            manifest = Path(directory) / "simulation.json"
            day = date(2026, 11, 19)
            self.assertEqual(simulate(client, state, manifest, day, 2, 5, send_not_before=send_time), 0)
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(simulate(client, saved, manifest, day, 2, 5), 0)

        self.assertEqual(client.batch.jobs.create.call_count, 2)
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[2][:2], ("Peux-tu détailler cet exemple ?", 1))
        self.assertTrue(all(item[2].startswith("synthetic-") for item in requests))
        self.assertEqual([item["outcome"] for item in saved["sessions"]], ["turn_limit", "user_finished"])

    def test_does_not_replay_an_interrupted_post(self):
        state = {"max_turns": 4, "sessions": [{"turns": [{"status": "sending"}]}]}
        client = MagicMock()
        with tempfile.TemporaryDirectory() as directory, patch(
            "utils.model_embedding.simulate_rag_users.requests.post"
        ) as post:
            manifest = Path(directory) / "simulation.json"
            self.assertEqual(simulate(client, state, manifest, date(2026, 11, 19), 2, 5), 1)
            post.assert_not_called()
        client.batch.jobs.create.assert_not_called()
        self.assertEqual(state["sessions"][0]["outcome"], "rag_error")
