from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class SpeakerVideo:
    video_id: int
    video_title: str = ""
    youtube_video_id: str = ""


@dataclass(frozen=True)
class SpeakerRecord:
    id: int
    name: str
    title: str | None
    videos: tuple[SpeakerVideo, ...] = ()


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
            key=lambda record: (normalize_text(record.name), record.id),
        )
        # Le poste ne participe jamais à la détection. Il ne sera examiné
        # qu'au moment de la revue du groupe construit depuis les noms.
        if len(grouped_records) > 1:
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
            speaker.name,
            speaker.title,
            video.id,
            video.title,
            video.youtube_video_id
        FROM speakers speaker
        LEFT JOIN video_speakers relation ON relation.speaker_id = speaker.id
        LEFT JOIN videos video ON video.id = relation.video_id
        ORDER BY speaker.id, video.id
        """
    )
    rows_by_speaker: dict[int, dict[str, object]] = {}
    for row in cursor.fetchall():
        speaker_id = int(row[0])
        item = rows_by_speaker.setdefault(
            speaker_id,
            {
                "name": str(row[1]).strip(),
                "title": (
                    (str(row[2]).strip() or None)
                    if row[2] is not None
                    else None
                ),
                "videos": [],
            },
        )
        if row[3] is not None:
            item["videos"].append(
                SpeakerVideo(
                    video_id=int(row[3]),
                    video_title=str(row[4] or "").strip(),
                    youtube_video_id=str(row[5] or "").strip(),
                )
            )
    return [
        SpeakerRecord(
            id=speaker_id,
            name=str(item["name"]),
            title=item["title"] if isinstance(item["title"], str) else None,
            videos=tuple(item["videos"]),
        )
        for speaker_id, item in rows_by_speaker.items()
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

    survivor = min(
        group.records,
        key=lambda record: (
            record.name != canonical_name,
            record.id,
        ),
    )
    duplicate_ids = [
        record.id for record in group.records if record.id != survivor.id
    ]
    video_ids = {
        video.video_id
        for record in group.records
        for video in record.videos
    }
    if duplicate_ids:
        cursor.execute(
            """
            INSERT INTO video_speakers (video_id, speaker_id, data_collected_date)
            SELECT relation.video_id, %s, now()
            FROM video_speakers relation
            WHERE relation.speaker_id = ANY(%s)
            ON CONFLICT (video_id, speaker_id) DO UPDATE SET
                data_collected_date = now()
            """,
            (survivor.id, duplicate_ids),
        )
        cursor.execute(
            "DELETE FROM speakers WHERE id = ANY(%s)",
            (duplicate_ids,),
        )
    cursor.execute(
        "UPDATE speakers SET name = %s, title = %s WHERE id = %s",
        (canonical_name, canonical_title, survivor.id),
    )

    return MergeResult(
        updated_videos=len(video_ids),
        deleted_rows=len(duplicate_ids),
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
        (video.youtube_video_id, video.video_title)
        for record in group.records
        for video in record.videos
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


def review_groups_tkinter(
    cursor,
    groups: Sequence[DuplicateGroup],
    *,
    output: Callable[[str], None] = print,
) -> tuple[int, int, int]:
    """Affiche les doublons un par un et applique les choix valides."""
    if not groups:
        return 0, 0, 0

    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter est requis pour revoir les doublons de speakers."
        ) from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(
            "Impossible d'ouvrir la fenêtre Tkinter de revue des speakers."
        ) from exc

    root.title("Fusion des speakers en doublon")
    root.geometry("900x620")
    root.minsize(760, 500)

    current_index = 0
    merged_groups = 0
    updated_videos = 0
    deleted_rows = 0

    progress_var = tk.StringVar()
    name_var = tk.StringVar()
    title_var = tk.StringVar()
    status_var = tk.StringVar(
        value="Choisis le nom et le poste à appliquer à toutes les occurrences."
    )

    container = ttk.Frame(root, padding=16)
    container.pack(fill="both", expand=True)
    container.columnconfigure(0, weight=1)
    container.rowconfigure(4, weight=1)

    ttk.Label(
        container,
        textvariable=progress_var,
        font=("Segoe UI", 13, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(0, 14))

    choices = ttk.Frame(container)
    choices.grid(row=1, column=0, sticky="ew")
    choices.columnconfigure(1, weight=1)

    ttk.Label(choices, text="Orthographe à conserver").grid(
        row=0, column=0, sticky="w", padx=(0, 12), pady=5
    )
    name_box = ttk.Combobox(
        choices,
        textvariable=name_var,
        state="readonly",
    )
    name_box.grid(row=0, column=1, sticky="ew", pady=5)

    ttk.Label(choices, text="Poste à conserver").grid(
        row=1, column=0, sticky="w", padx=(0, 12), pady=5
    )
    title_box = ttk.Combobox(choices, textvariable=title_var, state="normal")
    title_box.grid(row=1, column=1, sticky="ew", pady=5)

    ttk.Label(
        container,
        text=(
            "Le poste est éditable : sélectionne une valeur existante, "
            "écris-en une nouvelle, ou laisse le champ vide."
        ),
        foreground="#555555",
        wraplength=820,
    ).grid(row=2, column=0, sticky="w", pady=(6, 14))

    ttk.Label(container, text="Occurrences détectées").grid(
        row=3, column=0, sticky="w", pady=(0, 5)
    )
    occurrences = ttk.Treeview(
        container,
        columns=("name", "title", "video"),
        show="headings",
        selectmode="none",
    )
    occurrences.heading("name", text="Nom")
    occurrences.heading("title", text="Poste")
    occurrences.heading("video", text="Vidéo")
    occurrences.column("name", width=180, anchor="w")
    occurrences.column("title", width=230, anchor="w")
    occurrences.column("video", width=360, anchor="w")
    occurrences.grid(row=4, column=0, sticky="nsew")

    scrollbar = ttk.Scrollbar(
        container,
        orient="vertical",
        command=occurrences.yview,
    )
    occurrences.configure(yscrollcommand=scrollbar.set)
    scrollbar.grid(row=4, column=1, sticky="ns")

    ttk.Label(
        container,
        textvariable=status_var,
        wraplength=820,
    ).grid(row=5, column=0, sticky="w", pady=(12, 8))

    actions = ttk.Frame(container)
    actions.grid(row=6, column=0, sticky="e")

    def show_group() -> None:
        group = groups[current_index]
        progress_var.set(
            f"Doublon {current_index + 1} sur {len(groups)} "
            f"- {len(group.records)} occurrence(s)"
        )

        names = [str(value) for value in _preferred_values(group.names)]
        titles = [value or "" for value in _preferred_values(group.titles)]
        name_box.configure(values=names)
        title_box.configure(values=titles)
        name_var.set(names[0])
        title_var.set(titles[0] if titles else "")

        for item_id in occurrences.get_children():
            occurrences.delete(item_id)
        for record in group.records:
            video_labels = []
            for video in record.videos:
                label = video.youtube_video_id
                if video.video_title:
                    label = f"{label} - {video.video_title}"
                video_labels.append(label)
            occurrences.insert(
                "",
                "end",
                values=(record.name, record.title or "", " ; ".join(video_labels)),
            )

    def advance() -> None:
        nonlocal current_index
        current_index += 1
        if current_index >= len(groups):
            root.destroy()
            return
        show_group()

    def merge_current_group() -> None:
        nonlocal merged_groups, updated_videos, deleted_rows
        chosen_name = " ".join(name_var.get().split())
        chosen_title = " ".join(title_var.get().split()) or None
        if not chosen_name:
            messagebox.showwarning(
                "Nom obligatoire",
                "Choisis l'orthographe du nom à conserver.",
                parent=root,
            )
            return
        if not messagebox.askyesno(
            "Confirmer la fusion",
            (
                f"Appliquer le nom « {chosen_name} » et le poste "
                f"« {chosen_title or '(aucun poste)'} » à toutes les occurrences ?"
            ),
            parent=root,
        ):
            return

        result = merge_group(
            cursor,
            groups[current_index],
            canonical_name=chosen_name,
            canonical_title=chosen_title,
        )
        merged_groups += 1
        updated_videos += result.updated_videos
        deleted_rows += result.deleted_rows
        output(
            f"[merge] {chosen_name}: {result.updated_videos} vidéo(s) harmonisée(s), "
            f"{result.deleted_rows} ligne(s) supprimée(s)."
        )
        status_var.set(f"Fusion de {chosen_name} enregistrée.")
        advance()

    ttk.Button(actions, text="Terminer", command=root.destroy).pack(
        side="left", padx=(0, 8)
    )
    ttk.Button(actions, text="Passer", command=advance).pack(
        side="left", padx=(0, 8)
    )
    ttk.Button(
        actions,
        text="Fusionner et continuer",
        command=merge_current_group,
    ).pack(side="left")

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    show_group()
    root.mainloop()
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
            from pipeline.publish.sync_database import ensure_speakers_schema

            ensure_speakers_schema(cursor)
            records = fetch_speakers(cursor)
            groups = find_duplicate_groups(records, max_distance=max_distance)
            output(
                f"{len(records)} ligne(s) speaker analysee(s), "
                f"{len(groups)} groupe(s) a verifier."
            )
            if dry_run:
                result = review_groups(
                    cursor,
                    groups,
                    dry_run=True,
                    input_func=input_func,
                    output=output,
                )
                connection.rollback()
                output("Simulation terminee : aucune modification appliquee.")
            else:
                result = review_groups_tkinter(
                    cursor,
                    groups,
                    output=output,
                )
                connection.commit()
                output(
                    f"Termine : {result[0]} groupe(s) fusionne(s), "
                    f"{result[1]} video(s) harmonisee(s), "
                    f"{result[2]} ligne(s) supprimee(s)."
                )
            return result
