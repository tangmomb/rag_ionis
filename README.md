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
python -m pip install -r requirements-paddle-cu126.txt
```

Pour PaddleOCR GPU, la machine locale a ete verifiee avec un driver NVIDIA exposant CUDA 12.7 et un venv PyTorch en `cu126`. L'installation Paddle correspondante est:

```powershell
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install paddleocr
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
WHISPERX_MODEL=large-v3
WHISPERX_LANGUAGE=fr
WHISPERX_DEVICE=cuda
WHISPERX_COMPUTE_TYPE=float16
WHISPERX_BATCH_SIZE=16
YTDLP_FORMAT=bestvideo[height<=360]+bestaudio/best[height<=360]/best
YTDLP_MERGE_FORMAT=mp4
```

`TRANSCRIPT_SOURCE=auto` tente d'abord `youtube-transcript-api`, puis bascule vers Whisper si YouTube bloque ou si la transcription n'est pas disponible.

Demarrer la base:

```powershell
docker compose up -d postgres
```

Le volume Docker est nomme `rag_ionis_pgdata`. Il est independant du dossier projet et conserve les donnees entre les redemarrages/recreations du conteneur.

Le schema de base est initialise par un fichier unique:

```text
docker/postgres/init/001_schema.sql
```

Verifier la base:

```powershell
docker compose exec postgres psql -U rag_ionis -d rag_ionis -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

Tables principales:

- `videos`: videos de la chaine IONIS-STM, avec titre, lien et metadonnees stables.
- `video_daily_stats`: statistiques quotidiennes rattachees a une video via `video_id`, avec vues, likes et nombre de commentaires.
- `video_transcripts`: transcriptions rattachees a une video via `video_id`, avec une ligne par video/langue et les variantes `transcript`, `transcript_timecodes`, `transcript_timecodes_enrichi`.
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

## Run Pipeline

Executer toutes les steps dans l'ordre:

```powershell
.\.venv\Scripts\python.exe scripts/init/run_pipeline_init.py
.\.venv\Scripts\python.exe scripts/init/run_pipeline_init.py all
```

Sans argument, le script demande combien de videos traiter: `all` pour toute la chaine ou un nombre pour tester.

Tester le pipeline sur un nombre limite de videos:

```powershell
.\.venv\Scripts\python.exe scripts/init/run_pipeline_init.py 3
```

Le script lance les steps 00 a 14. Au demarrage, il vide les tables applicatives SQL en conservant le schema, puis supprime les anciens dossiers locaux `*_init` dans `downloads/youtube/`. La Step 02 cree ensuite un nouveau dossier date suffixe `_init`, puis ce meme dossier est passe aux steps suivantes.

Un run d'initialisation remplace le precedent:

- localement, `downloads/youtube/` ne conserve qu'un seul dossier `*_init`;
- dans S3, les anciens prefixes `youtube/*_init` sont supprimes avant le nouvel upload;
- dans SQL, les donnees applicatives sont videes avant la recollecte, tout en gardant l'architecture de tables.

Options utiles:

```powershell
.\.venv\Scripts\python.exe scripts/init/run_pipeline_init.py 3 --dry-run-upload --dry-run-sql
.\.venv\Scripts\python.exe scripts/init/run_pipeline_init.py 3 --skip-upload
.\.venv\Scripts\python.exe scripts/init/run_pipeline_init.py all --skip-data
.\.venv\Scripts\python.exe scripts/init/run_pipeline_init.py 3 --force
```

## Step 02 - Download Videos

Telecharger en 360p les videos referencees dans la table SQL `videos`:

```powershell
python scripts/init/02_download_videos.py
```

Chaque lancement cree un sous-dossier date suffixe `_init` dans `downloads/youtube/`, puis un dossier par video:

```text
downloads/youtube/20260628_1312_init/
  LJ-W6BjSJRo/
    LJ-W6BjSJRo.mp4
```

## Step 06 - Whisper Transcription

Transcrire localement avec `whisperx` les videos du dernier dossier de telechargement, sauf si des sous-titres OCR sont deja presents:

```powershell
python scripts/init/07_whisper_transcription.py
```

Le script extrait un fichier audio temporaire avec ffmpeg, transcrit localement avec `whisperx` en francais sur GPU, puis aligne les segments pour produire des timecodes. Il cree un dossier `transcript/` dans chaque dossier video et produit un fichier horodate, par exemple `hGUkhjssd_transcript_timecodes.txt`.

Les parametres utiles se reglant via `.env` sont `WHISPERX_MODEL`, `WHISPERX_LANGUAGE`, `WHISPERX_DEVICE`, `WHISPERX_COMPUTE_TYPE` et `WHISPERX_BATCH_SIZE`.

Pour forcer un usage GPU, garde `WHISPERX_DEVICE=cuda` et `WHISPERX_COMPUTE_TYPE=float16`. Pour des machines plus legeres, `WHISPERX_DEVICE=cpu` et `WHISPERX_COMPUTE_TYPE=int8` restent possibles. Le modele par defaut est maintenant `large-v3`.

Quand la transcription produit des timecodes, le fichier se termine par `_transcript_timecodes.txt`.

## Step 03 - Extract Images

Extraire une image toutes les 2 secondes pour chaque video:

```powershell
python scripts/init/04_extract_images.py
```

Le script cree un dossier `images/` dans chaque dossier video. Les images sont nommees par timecode minute/seconde, par exemple `00_00.jpg`, `00_02.jpg`, `01_00.jpg`.

## Step 04 - Image OCR

Extraire localement les textes visibles avec PaddleOCR sur toutes les images:

```powershell
python scripts/init/05_images_ocr.py
```

Le script lit toutes les images dans `images/` et ecrit `transcript/<video_id>_ocr_processed.json`. Par defaut, seules les detections OCR avec `rec_score >= 0.9` sont conservees dans le JSON traite, afin d'eviter les textes de decor peu fiables. Le brut est conservé en `transcript/<video_id>_ocr_brut.json`.

## Step 05 - OCR Subtitles

Concatener les items OCR de type `subtitle` dans un fichier texte dedie, avec une version timecodee en parallele:

```powershell
python scripts/init/06_ocr_subtitles.py
```

Le script lit `transcript/<video_id>_ocr_processed.json` et ecrit `transcript/<video_id>_ocr_subtitle.txt` ainsi que `transcript/<video_id>_ocr_subtitle_timecodes.txt`.

## Step 07 - Enrich Transcripts

Ajouter les textes visibles a l'ecran dans les transcripts timecodes:

```powershell
python scripts/init/08_enrich_transcripts.py
```

Le script n'appelle aucune API. Il combine `transcript/*_transcript_timecodes.txt` avec `transcript/*_ocr_processed.json` et cree `transcript/*_transcript_timecodes_enrichi.txt`.

## Step 13 - Upload Videos To S3

Uploader le dernier dossier de videos vers le bucket S3 en conservant la meme arborescence:

```powershell
python scripts/init/08_upload_videos_to_s3.py
```

Configuration requise dans `.env`:

```powershell
S3_BUCKET_NAME=rag-ionis-532523613357-eu-west-3-an
S3_REGION=eu-west-3
S3_ACCESS_KEY_ID=votre_access_key
S3_SECRET_ACCESS_KEY=votre_secret_key
```

Par defaut, le script prend le dernier dossier de `downloads/youtube/` et l'upload dans le prefixe S3 `youtube/`:

```text
downloads/youtube/20260628_1312_init/LJ-W6BjSJRo/LJ-W6BjSJRo.mp4
-> s3://bucket/youtube/20260628_1312_init/LJ-W6BjSJRo/LJ-W6BjSJRo.mp4
```

Options utiles:

```powershell
python scripts/init/08_upload_videos_to_s3.py --dry-run
python scripts/init/08_upload_videos_to_s3.py --video-dir downloads/youtube/20260628_1312_init
python scripts/init/08_upload_videos_to_s3.py --prefix youtube/20260628_1312_init
python scripts/init/08_upload_videos_to_s3.py --clean-init-prefix
python scripts/init/08_upload_videos_to_s3.py --force
```

## Step 14 - Update SQL Assets

Mettre a jour la base SQL avec les chemins S3 des fichiers generes et synchroniser les transcripts disponibles:

```powershell
python scripts/init/09_update_sql_assets.py
```

Le script cree la table `video_elements` si elle n'existe pas, puis y enregistre les videos, images, analyses et transcripts du dernier dossier de `downloads/youtube/`. Il utilise le meme prefixe S3 `youtube/` que la Step 08 par defaut:

```text
downloads/youtube/20260628_1312_init/LJ-W6BjSJRo/LJ-W6BjSJRo.mp4
-> s3://bucket/youtube/20260628_1312_init/LJ-W6BjSJRo/LJ-W6BjSJRo.mp4
```

Il met aussi a jour une seule ligne `video_transcripts` par video/langue avec les trois variantes trouvees dans chaque dossier `transcript/`:

```text
transcript                  -> *_transcript.txt
transcript_timecodes        -> *_transcript_timecodes.txt
transcript_timecodes_enrichi -> *_transcript_timecodes_enrichi.txt
```

Options utiles:

```powershell
python scripts/init/09_update_sql_assets.py --dry-run
python scripts/init/09_update_sql_assets.py --video-dir downloads/youtube/20260628_1312_init
python scripts/init/09_update_sql_assets.py --prefix youtube/20260628_1312_init
python scripts/init/09_update_sql_assets.py --clean-init-assets
python scripts/init/09_update_sql_assets.py --skip-transcripts
```

## Step 99 - Clear Database

Recreer la base SQL depuis le schema. Attention: cette commande supprime toutes les donnees des tables applicatives avant de recharger `docker/postgres/init/001_schema.sql`.

```powershell
python utils/reset_database.py
```

Vider les tables applicatives sans supprimer le schema:

```powershell
python utils/99_clear_database.py
```

## Outils locaux

Console HTML pour tester les requetes YouTube API:

```text
utils/api_console.html
```
