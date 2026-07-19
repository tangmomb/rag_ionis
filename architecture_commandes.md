# Architecture du projet et commandes

## Vue d'ensemble

```text
Vidéo → Inspection → Choix du chemin → Nettoyage des textes → Transcription
       → Speakers → Chunks → Embeddings → Base de données → Recherche RAG
```

Toutes les commandes se lancent depuis `E:\cooking\rag_ionis` avec le Python du projet :

```powershell
cd E:\cooking\rag_ionis
.\.venv\Scripts\python.exe
```

## 1. Récupérer les informations YouTube

```powershell
.\.venv\Scripts\python.exe -m pipeline.ingest.fetch_youtube_metadata
```

Appelle `pipeline/ingest/fetch_youtube_metadata.py`. Le script contacte l'API
YouTube et récupère les identifiants, titres, descriptions, durées, dates, URLs
et statistiques. Il recrée le cache dans
`downloads/youtube/info_videos/`, puis écrase aussi
`downloads/youtube/init/VIDEO_ID/metadata/youtube_video_metadata.json` pour
chaque vidéo déjà présente. Les fichiers vidéo ne sont ni modifiés ni
retéléchargés par cette commande.

## 2. Télécharger les vidéos

```powershell
.\.venv\Scripts\python.exe -m pipeline.ingest.download_videos
```

Appelle `pipeline/ingest/download_videos.py`, qui utilise `yt-dlp` et `FFmpeg`. Les vidéos sont rangées dans `downloads/youtube/init/VIDEO_ID/`, par exemple `VIDEO_ID/VIDEO_ID.mp4`.

À ce stade, la vidéo est téléchargée mais pas encore analysée.

## 3. Examiner techniquement la vidéo

```powershell
.\.venv\Scripts\python.exe -m pipeline inspect VIDEO_ID --probe-only
```

Le chemin d'appel, avec les fonctions indiquées, est :

```text
pipeline/__main__.py : main()
→ pipeline/cli.py : main()
→ pipeline/orchestrator.py : inspect_video(probe_only=True)
→ pipeline/context.py : PipelineContext.inspect()
→ pipeline/probe.py : probe_video()
→ retour dans orchestrator.py : arrêt si probe_only=True
```

Le projet vérifie la durée, la résolution, les codecs, les FPS, l'audio et la rotation. Le résultat est écrit dans `VIDEO_ID/metadata/video_manifest.json`.

`--probe-only` lit uniquement les caractéristiques techniques.

## 4. Inspecter le contenu de la vidéo

```powershell
.\.venv\Scripts\python.exe -m pipeline inspect VIDEO_ID
```

Le chemin d'appel, avec les fonctions indiquées, est :

```text
pipeline/__main__.py : main()
→ pipeline/cli.py : main()
→ pipeline/orchestrator.py : inspect_video(probe_only=False)
→ pipeline/context.py : PipelineContext.inspect()
→ pipeline/probe.py : probe_video()
→ pipeline/planner.py : inspection_plan()
→ pipeline/executor.py : execute_tasks()
→ pipeline/step_handlers.py : handler de chaque tâche
→ pipeline/steps/inspection/*.py : fonctions métier
```

Les sept tâches d'inspection sont :

```text
frames.extract
frames.classify
video.detect_interview
video.infer_type
ocr.extract_raw
ocr.extract_boxes
video.detect_subtitles
```

Les handlers utilisent notamment les fichiers et fonctions suivants :

```text
pipeline/steps/inspection/extract_frames.py : extract_images()
pipeline/steps/inspection/classify_frames.py : classify_video()
pipeline/steps/inspection/detect_interviews.py : detect_video()
pipeline/steps/inspection/infer_video_type.py : infer_for_video()
pipeline/steps/inspection/extract_raw_ocr.py : extract_for_video_isolated()
pipeline/steps/inspection/extract_ocr_boxes.py : extract_for_video()
pipeline/steps/inspection/detect_subtitles.py : détection des sous-titres
```

Les sorties sont placées dans `outputs/images/`, `outputs/interview/` et `outputs/ocr/`. Le manifeste mémorise ensuite le type de vidéo et la présence de sous-titres :

```json
{
  "routing_facts": {
    "has_subtitles": true,
    "video_type": "interview"
  }
}
```

## 5. Voir le plan choisi

```powershell
.\.venv\Scripts\python.exe -m pipeline plan VIDEO_ID
```

Cette commande appelle `pipeline/planner.py : processing_plan()`. Le planner choisit la suite selon le type, les sous-titres et la durée.

Pour une vidéo courte avec sous-titres, WhisperX reste la chaîne canonique et la
référence OCR de correction est construite :

```text
ocr.build_processed → ocr.filter_overlays → ocr.extract_review_candidates
→ ocr.review_other_text → ocr.apply_review
→ transcript.whisper
→ transcript.extract_ocr → transcript.normalize_brand
→ transcript.create_plain_ocr → transcript.reconcile_ocr
→ speakers.propose → speakers.validate → transcript.apply_speakers
→ transcript.enrich → transcript.create_plain
→ chunks.create → embeddings.create
```

On obtient alors le WhisperX brut, l'unique
`transcripts_ocr/plain_transcript.txt` réservé aux corrections et le WhisperX
corrigé par `gpt-5.6-luna` après rapprochement. Luna renvoie uniquement le texte
corrigé de chaque segment ; Python préserve les timecodes, les labels de speaker,
l'ordre et le nombre de segments. Les intermédiaires OCR timecodés restent dans
`outputs/ocr/`. Pour une vidéo sans sous-titres,
`transcript.correct_whisper` remplace `transcript.reconcile_ocr` et les tâches
OCR de correction de sous-titres ne sont pas ajoutées.

Cette commande prépare le plan mais n'exécute pas les traitements.

## 6. Exécuter tout le traitement

```powershell
.\.venv\Scripts\python.exe -m pipeline run VIDEO_ID
```

C'est la commande normale de bout en bout. Elle appelle :

```text
pipeline/__main__.py : main()
→ pipeline/cli.py : main()
→ pipeline/orchestrator.py : run_video()
→ pipeline/context.py : PipelineContext.inspect()
→ pipeline/planner.py : inspection_plan() puis processing_plan()
→ pipeline/executor.py : execute_tasks()
→ pipeline/step_handlers.py : handlers des tâches
→ pipeline/steps/ : fonctions métier
```

Elle enchaîne l'inspection, le nettoyage OCR, le transcript WhisperX canonique,
l'éventuelle référence OCR de correction, l'identification des speakers,
l'ajout des seuls intercalaires dans `transcript_enriched`, les chunks et les
embeddings.

Les speakers validés restent un attribut de la vidéo et du transcript avec
speakers. Ils ne sont plus copiés dans les chunks JSON ni dans la table SQL
`chunks`.

Le planner décide des étapes dans `pipeline/planner.py`, l'executor les exécute dans `pipeline/executor.py`, et `pipeline/catalog.py` relie chaque identifiant à sa fonction Python. Par exemple :

```text
frames.extract → step_handlers.extract_frames()
→ steps.inspection.extract_frames.extract_images()
```

## 7. Les résultats produits

```text
VIDEO_ID/
├── VIDEO_ID.mp4
├── metadata/
│   ├── youtube_video_metadata.json
│   └── video_manifest.json
└── outputs/
    ├── images/
    ├── interview/
    ├── ocr/
    ├── speakers/
    ├── transcripts_whisper/  # source canonique
    ├── transcripts_ocr/      # uniquement plain_transcript.txt pour correction
    ├── chunks/
    └── embeddings/
```

Le manifeste conserve l'état et les fichiers produits par chaque tâche.

## 8. Relancer ou simuler

Réutiliser les décisions déjà présentes dans le manifeste :

```powershell
.\.venv\Scripts\python.exe -m pipeline run VIDEO_ID --skip-inspection
```

Tout reconstruire :

```powershell
.\.venv\Scripts\python.exe -m pipeline run VIDEO_ID --force
```

Cette variante supprime entièrement le dossier `outputs/` de chaque vidéo
sélectionnée avant de relancer l'inspection et le traitement. Les vidéos et le
dossier `metadata/` sont conservés. `pipeline task ... --force` reste ciblé et
ne supprime pas `outputs/`.

Simuler sans exécuter :

```powershell
.\.venv\Scripts\python.exe -m pipeline run VIDEO_ID --dry-run
```

Le système utilise les checkpoints de `metadata/video_manifest.json`. Une étape dont les sorties existent encore peut être marquée `cached` et ne pas être relancée.

## 9. Exécuter une seule étape

Afficher le catalogue :

```powershell
.\.venv\Scripts\python.exe -m pipeline task --list
```

Extraire uniquement les images :

```powershell
.\.venv\Scripts\python.exe -m pipeline task frames.extract VIDEO_ID
```

Le chemin est `pipeline/cli.py → PipelineContext.inspect() → catalog.TASKS["frames.extract"] → step_handlers.extract_frames() → pipeline/steps/inspection/extract_frames.py`.

Lancer uniquement WhisperX :

```powershell
.\.venv\Scripts\python.exe -m pipeline task transcript.whisper VIDEO_ID
```

Créer uniquement les chunks :

```powershell
.\.venv\Scripts\python.exe -m pipeline task chunks.create VIDEO_ID
```

## 10. Envoyer les résultats vers le stockage et la base

Démarrer PostgreSQL et Phoenix :

```powershell
docker compose up -d postgres phoenix
```

Uploader la vidéo et les sorties :

```powershell
.\.venv\Scripts\python.exe -m pipeline.publish.upload_outputs_to_s3
```

Appelle `pipeline/publish/upload_outputs_to_s3.py`.

Synchroniser les métadonnées, transcripts et chunks avec PostgreSQL :

```powershell
.\.venv\Scripts\python.exe -m pipeline.publish.sync_database
```

Appelle `pipeline/publish/sync_database.py`.

## 11. Lancer l'interface RAG

```powershell
.\.venv\Scripts\python.exe -m uvicorn interface.app:app --host 127.0.0.1 --port 8000
```

Cette commande démarre `interface/app.py`. L'interface utilise ensuite `interface/backend/orchestration.py`, `planner.py`, `retrieval.py` et `generation.py`.

Lorsqu'une question est posée :

```text
interface.app → orchestration → planner → retrieval
→ PostgreSQL / recherche vectorielle / BM25 → generation → réponse finale
```

## Parcours complet des commandes

```powershell
docker compose up -d postgres phoenix
.\.venv\Scripts\python.exe -m pipeline.ingest.fetch_youtube_metadata
.\.venv\Scripts\python.exe -m pipeline.ingest.download_videos
.\.venv\Scripts\python.exe -m pipeline run VIDEO_ID
.\.venv\Scripts\python.exe -m pipeline.publish.upload_outputs_to_s3
.\.venv\Scripts\python.exe -m pipeline.publish.sync_database
.\.venv\Scripts\python.exe -m uvicorn interface.app:app --host 127.0.0.1 --port 8000
```

En résumé, `python -m pipeline run VIDEO_ID` est la grande commande centrale : elle relie automatiquement l'inspection, le choix de la route, la transcription, le découpage et les embeddings.
