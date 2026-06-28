# RAG IONIS

Base locale pour construire un RAG.

## Environnement Python

Le venv est dans `.venv` et contient deja PyTorch CUDA et Whisper.

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
```

Verifications utiles:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
whisper --help
```

Reinstallation equivalente:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-torch-cu126.txt
python -m pip install -r requirements.txt
```

## Postgres + pgvector

Copier la configuration locale:

```powershell
Copy-Item .env.example .env
```

Renseigner ensuite la cle API YouTube dans `.env`:

```powershell
YOUTUBE_API_KEY=votre_cle_api
OPENAI_API_KEY=votre_cle_api_openai
```

Les transcriptions utilisent une source non officielle et peuvent etre bloquees par YouTube. Ces variables permettent de ralentir les tentatives:

```powershell
YOUTUBE_API_SLEEP_SECONDS=2
TRANSCRIPT_RETRIES=2
TRANSCRIPT_RETRY_SECONDS=10
TRANSCRIPT_SLEEP_SECONDS=5
```

Par defaut, le script transcrit avec `yt-dlp` + Whisper: il telecharge la video en 360p dans `downloads/youtube/`, puis transcrit le fichier localement.

```powershell
TRANSCRIPT_SOURCE=whisper
WHISPER_MODEL=small
WHISPER_LANGUAGE=fr
OPENAI_TRANSCRIBE_MODEL=whisper-1
OPENAI_TRANSCRIBE_LANGUAGE=fr
OPENAI_TRANSCRIBE_AUDIO_BITRATE=48k
YTDLP_FORMAT=bestvideo[height<=360]+bestaudio/best[height<=360]/best
YTDLP_MERGE_FORMAT=mp4
```

`TRANSCRIPT_SOURCE=auto` tente d'abord `youtube-transcript-api`, puis bascule vers Whisper si YouTube bloque ou si la transcription n'est pas disponible.

Demarrer la base:

```powershell
docker compose up -d postgres
```

Le volume Docker est nomme `rag_ionis_pgdata`. Il est independant du dossier projet et conserve les donnees entre les redemarrages/recreations du conteneur.

Verifier la base:

```powershell
docker compose exec postgres psql -U rag_ionis -d rag_ionis -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

Tables principales:

- `videos`: videos de la chaine IONIS-STM, avec titre, lien et metadonnees stables.
- `video_daily_stats`: statistiques quotidiennes rattachees a une video via `video_id`, avec vues, likes et nombre de commentaires.
- `video_transcripts`: transcriptions rattachees a une video via `video_id`, avec texte complet et segments horodates.
- `comments`: commentaires rattaches a une video via `video_id`, avec support des reponses via `parent_comment_id`.

Comparer les vues entre deux jours:

```sql
SELECT
    v.title,
    newer.view_count - older.view_count AS views_delta
FROM videos v
JOIN video_daily_stats older ON older.video_id = v.id
JOIN video_daily_stats newer ON newer.video_id = v.id
WHERE older.snapshot_date = '2026-06-25'
  AND newer.snapshot_date = '2026-06-26'
ORDER BY views_delta DESC;
```

Arreter sans supprimer les donnees:

```powershell
docker compose down
```

Supprimer aussi les donnees:

```powershell
docker volume rm rag_ionis_pgdata
```

## Step 01 - Get Data

Importer les videos de la chaine, leurs statistiques du jour et leurs commentaires:

```powershell
python scripts/init/01_get_data.py
```

Le script travaille sur `https://www.youtube.com/@IONIS-STM/videos` et remplit les tables `videos`, `video_daily_stats`, `video_transcripts` et `comments`.

## Step 02 - Download Videos

Telecharger en 360p les videos referencees dans la table SQL `videos`:

```powershell
python scripts/init/02_download_videos.py
```

Chaque lancement cree un sous-dossier date dans `downloads/youtube/`, par exemple `downloads/youtube/20260628_1312/`.

## Step 03 - Transcribe Videos

Transcrire avec l'API OpenAI les videos du dernier dossier de telechargement:

```powershell
python scripts/init/03_transcribe_videos.py
```

Le script extrait un fichier audio temporaire avec ffmpeg, appelle `whisper-1` en francais, puis cree un dossier `transcript/` a cote des videos et produit un fichier horodate par video, par exemple `hGUkhjssd_transcript_timecodes.txt`.

Pour detecter les intervenants, utiliser `OPENAI_TRANSCRIBE_MODEL=gpt-4o-transcribe-diarize`. Ce modele n'accepte pas de prompt de guidage et peut moins bien respecter le francais sur ce corpus.

Quand le modele produit des timecodes (`whisper-1` ou `gpt-4o-transcribe-diarize`), le fichier se termine par `_transcript_timecodes.txt`.

## Step 04 - Strip Timecodes

Produire les fichiers de transcription sans timecodes a partir des fichiers `*_transcript_timecodes.txt`:

```powershell
python scripts/init/04_strip_timecodes.py
```

Le script n'appelle aucune API. Il cree les fichiers freres `*_transcript.txt`.

## Step 05 - Extract Images

Extraire une image toutes les 2 secondes pour chaque video:

```powershell
python scripts/init/05_extract_images.py
```

Le script cree un dossier `images/` a cote de `transcript/`, puis un sous-dossier par video. Les images sont nommees par timecode minute/seconde, par exemple `00_00.jpg`, `00_02.jpg`, `01_00.jpg`.

## Step 06 - Analyze Image Text

Detecter les images ou du texte ecrit apparait a l'ecran:

```powershell
python scripts/init/06_analyze_image_text.py
```

Le script envoie les images en `detail: low` au modele `OPENAI_IMAGE_ANALYZE_MODEL` (`gpt-5.4-nano` par defaut) et ecrit les resultats dans `images/analyse/`.

## Step 99 - Clear Database

Vider les tables applicatives sans supprimer le schema:

```powershell
python utils/99_clear_database.py
```

## Outils locaux

Console HTML pour tester les requetes YouTube API:

```text
utils/api_console.html
```
