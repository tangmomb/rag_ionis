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
- `stats`: statistiques quotidiennes rattachees a une video via `video_id`, avec vues, likes et nombre de commentaires.
- `transcripts`: transcriptions rattachees a une video via `video_id`, avec une ligne par video/langue et les variantes `transcript`, `transcript_timecodes`, `transcript_timecodes_enrichi`.
- `chunks`: chunks textuels rattaches a une video via `video_id`, avec contenu, speakers, alertes et embedding quand il existe.
- `comments`: commentaires rattaches a une video via `video_id`, avec support des reponses via `parent_comment_id`.

Comparer les vues entre deux jours:

```sql
SELECT
    v.title,
    newer.view_count - older.view_count AS views_delta
FROM videos v
JOIN stats older ON older.video_id = v.id
JOIN stats newer ON newer.video_id = v.id
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

Le script travaille sur `https://www.youtube.com/@IONIS-STM/videos` et remplit les tables `videos`, `stats`, `transcripts` et `comments`.

## Run Pipeline

Executer toutes les steps dans l'ordre:

```powershell
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py all
```

Sans argument, le script demande quoi traiter: `all` pour toute la chaine, un nombre pour tester, ou un lien YouTube precis.

Tester le pipeline sur un nombre limite de videos:

```powershell
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py 3
```

Tester le pipeline sur une video precise:

```powershell
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Au demarrage, le script vide les tables applicatives SQL en conservant le schema, puis supprime les anciens dossiers locaux `*_init` dans `downloads/youtube/`. Il lance ensuite les steps 01 a 26. La Step 02 cree un nouveau dossier date suffixe `_init`, puis ce meme dossier est passe aux steps suivantes.

Un run d'initialisation remplace le precedent:

- localement, `downloads/youtube/` ne conserve qu'un seul dossier `*_init`;
- dans S3, les anciens prefixes `youtube/*_init` sont supprimes avant le nouvel upload;
- dans SQL, les donnees applicatives sont videes avant la recollecte, tout en gardant l'architecture de tables.

Options utiles:

```powershell
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py 3 --dry-run-upload --dry-run-sql
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py 3 --skip-upload
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py all --skip-data
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py 3 --force
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
    metadata/
      youtube_video_metadata.json
      pipeline_analysis.json
    outputs/
      images/
      interview/
      ocr/
      transcripts/
      chunks/
```

Le fichier video reste a la racine du dossier video. Les metadonnees vont dans `metadata/`, et tous les fichiers generes par le pipeline vont dans `outputs/` pour eviter de melanger frames, OCR, transcripts et chunks.

## Step 03 - Extract Images

Extraire une image toutes les 2 secondes pour chaque video:

```powershell
python scripts/init/03_extract_images.py
```

Le script cree `outputs/images/` dans chaque dossier video. Les images sont nommees par timecode minute/seconde, par exemple `00_00.jpg`, `00_02.jpg`, `01_00.jpg`.

## Step 04 - Classify Images

Classer localement les images extraites avec le modele DINO+CLIP entraine:

```powershell
python scripts/init/04_classify_images.py
```

Le script lit `outputs/images/`, calcule les embeddings attendus par le modele, appelle le classifieur, puis reorganise directement les fichiers selon les trois sorties du modele: `outputs/images/footage/`, `outputs/images/graphic/` et `outputs/images/mixture/`. Il ne lance pas de k-means et ne cree pas de sous-dossiers `graphic_XX`.

Il ecrit aussi `outputs/images/frame_classification_manifest.json` avec le label predit de chaque image et `outputs/images/frame_classification_features.json` avec les memes predictions pour compatibilite avec les etapes suivantes. La Step 07 OCR lit ensuite les images recursivement et conserve ces chemins relatifs dans ses JSON.

Options utiles:

```powershell
python scripts/init/04_classify_images.py --model-path models/frame_filter_2026-07-02_21-30-31.joblib
python scripts/init/04_classify_images.py --batch-size 16
python scripts/init/04_classify_images.py --device cuda
python scripts/init/04_classify_images.py --force
```

## Step 05 - Detect Interviews

Detecter les videos de type interview a partir des frames `footage/` consecutives:

```powershell
python scripts/init/05_detect_interviews.py
```

Le script ecrit un manifeste `outputs/interview/interview_detection_manifest.json`, utilise ensuite par l'inference du type de video.

## Step 06 - Infer Video Type

Deduir le type de video depuis le manifeste de classification images:

```powershell
python scripts/init/06_infer_video_type.py
```

Le script ecrit `video_type` dans `metadata/pipeline_analysis.json`.

## Step 07 - Extract Raw OCR

Extraire localement l'OCR brut avec PaddleOCR sur toutes les images:

```powershell
python scripts/init/07_extract_raw_ocr.py
```

Le script lit toutes les images dans `outputs/images/` et ecrit les JSON OCR bruts dans `outputs/ocr/raw_ocr_<group>_frames.json`. Il ne produit plus directement le fichier processed.

## Step 08 - Extract OCR Boxes

Extraire les emplacements OCR depuis le JSON brut, sans refaire tourner PaddleOCR:

```powershell
python scripts/init/08_extract_ocr_boxes.py
```

Le script lit les JSON OCR bruts dans `outputs/ocr/` et ecrit `outputs/ocr/ocr_box_locations.json`.

## Step 09 - Detect OCR Subtitles

Detecter si la video contient probablement des sous-titres OCR a partir des boxes:

```powershell
python scripts/init/09_detect_ocr_subtitles.py
```

Le script inspecte les boxes en bas de video, au centre, et verifie qu'un centre approximatif reste present de facon continue pendant au moins 10 secondes. Il ecrit `has_subtitles` dans `metadata/pipeline_analysis.json`.

## Step 10 - Build Processed OCR

Transformer l'OCR brut en OCR traite sans relancer PaddleOCR:

```powershell
python scripts/init/10_build_processed_ocr.py
```

Le script lit les JSON bruts de `outputs/ocr/` et ecrit `outputs/ocr/processed_ocr_items.json`. Par defaut, seules les detections OCR avec `rec_score >= 0.9` sont conservees dans le JSON traite, puis les textes de decor probables sont filtres par taille, isolement, persistance statique avec variantes OCR proches, fragments progressifs et liste d'exclusion legere. Les detections provenant de `outputs/images/graphic/` sont conservees avec `kind: "graphic"`. Pour les detections non sous-titres venant de `outputs/images/footage/` ou `outputs/images/mixture/`, un meme mot ou une meme phrase repete dans une fenetre de 20 frames ne garde que sa derniere occurrence. Les textes non sous-titres finissant par `?` sont classes comme `question_intertitle`. Les sous-titres OCR sont detectes par une ligne de position relative recurrente, principalement le centre X commun des boites, avec des garde-fous geometriques sur Y, largeur et hauteur.

## Step 11 - Filter Processed OCR

Filtrer et consolider les items OCR traites:

```powershell
python scripts/init/11_filter_processed_ocr.py
```

Le script lit `outputs/ocr/corrected_ocr_items.json` quand il existe, sinon `outputs/ocr/processed_ocr_items.json`, puis ecrit `outputs/ocr/filtered_ocr_overlays.json`.

## Step 12 - Extract Other Text Review Candidates

Exporter les images entieres des items `kind: "others"` depuis le JSON filtered, avec une box rouge autour de la zone OCR:

```powershell
python scripts/init/12_extract_other_text_review_candidates.py
```

Le script lit `outputs/ocr/filtered_ocr_overlays.json`, recharge l'image source complete, dessine une box rouge autour de chaque zone `kinds_details.others` avec un padding de `5 px`, ecrit ces images annotees dans `outputs/ocr/other_text_review_candidates/`, puis ajoute un manifeste `outputs/ocr/other_text_review_candidates/review_candidates_manifest.json`.

## Step 13 - Review Other Text Candidates

Demander a `gpt-5.4-nano` si le texte dans la zone rouge des images `others` ressemble a du texte ajoute au montage:

```powershell
python scripts/init/13_review_other_text_candidates.py
```

Le script lit `outputs/ocr/other_text_review_candidates/review_candidates_manifest.json`, envoie chaque image annotee a OpenAI avec la consigne de se concentrer sur la zone encadree en rouge, puis ecrit les prompts, reponses et decisions parsees dans `outputs/ocr/other_text_gpt_review/`, avec un resume global dans `review_summary.json`.

## Step 14 - Apply Other Text Review

Appliquer le `review_summary.json` de review GPT pour produire une nouvelle version du filtered sans ecraser l'original :

```bash
python scripts/init/14_apply_other_text_review.py
```

Le script lit `outputs/ocr/filtered_ocr_overlays.json` et `outputs/ocr/other_text_gpt_review/review_summary.json`, puis ecrit `outputs/ocr/reviewed_ocr_overlays.json`. Les items `others` avec `is_added_in_edit: false` sont supprimes, et ceux avec `has_ocr_error: true` voient leur texte remplace par `corrected_text`.

## Step 15 - OCR Subtitles

Concatener les items OCR de type `subtitle` dans un fichier texte dedie, avec une version timecodee en parallele:

```powershell
python scripts/init/15_extract_ocr_subtitles.py
```

Le script ne traite une video que si `metadata/pipeline_analysis.json` contient `has_subtitles: true`. Il lit alors `outputs/ocr/corrected_ocr_items.json` quand il existe, sinon `outputs/ocr/processed_ocr_items.json`, et ecrit `outputs/transcripts/ocr_subtitles.txt` ainsi que `outputs/transcripts/ocr_subtitles_timecoded.txt`.

## Step 16 - Transcribe With Whisper

Transcrire localement avec `whisperx` les videos du dernier dossier de telechargement uniquement si `metadata/pipeline_analysis.json` contient `has_subtitles: false`:

```powershell
python scripts/init/16_transcribe_with_whisper.py
```

Le script extrait un fichier audio temporaire avec ffmpeg, transcrit localement avec `whisperx` en francais sur GPU, puis aligne les segments pour produire des timecodes. Il ecrit `outputs/transcripts/whisper_transcript_timecoded.txt`.

Les parametres utiles se reglant via `.env` sont `WHISPERX_MODEL`, `WHISPERX_LANGUAGE`, `WHISPERX_DEVICE`, `WHISPERX_COMPUTE_TYPE` et `WHISPERX_BATCH_SIZE`.

Pour forcer un usage GPU, garde `WHISPERX_DEVICE=cuda` et `WHISPERX_COMPUTE_TYPE=float16`. Pour des machines plus legeres, `WHISPERX_DEVICE=cpu` et `WHISPERX_COMPUTE_TYPE=int8` restent possibles. Le modele par defaut est maintenant `large-v3`.

Quand la transcription produit des timecodes, le fichier attendu par les steps suivantes est `whisper_transcript_timecoded.txt`.

## Step 17 - Correct Transcript Timecodes

Corriger certains mots du transcript timecode en les comparant aux mots OCR trouves dans `processed_ocr_items.json`, pour essayer de recuperer des noms propres visibles a l'ecran:

```powershell
python scripts/init/17_correct_transcript_timecodes.py
```

Le script lit `outputs/transcripts/whisper_transcript_timecoded.txt` ou `outputs/transcripts/ocr_subtitles_timecoded.txt`, puis ecrit `whisper_transcript_timecoded_corrected.txt` ou `ocr_subtitles_timecoded_corrected.txt`. Il conserve la casse reelle vue par l'OCR et se concentre sur les zones `name`, `lower_third`, `title`, `logo` et `graphic` pour limiter les faux positifs.

Le niveau de correction est ajustable:

```powershell
python scripts/init/17_correct_transcript_timecodes.py --mode conservative
python scripts/init/17_correct_transcript_timecodes.py --mode balanced
python scripts/init/17_correct_transcript_timecodes.py --mode aggressive
```

`conservative` corrige peu, `aggressive` accepte plus de noms proches, et `balanced` est le defaut.

## Step 18 - Enrich Timecodes

Ajouter les textes visibles a l'ecran dans les timecodes corriges:

```powershell
python scripts/init/18_enrich_transcripts.py
```

Le script n'appelle aucune API. Il combine les fichiers corriges avec les overlays filtres de `outputs/ocr/`, et ajoute le suffixe `_enriched.txt` au fichier source. Un fichier `whisper_transcript_timecoded_corrected.txt` produit donc `whisper_transcript_timecoded_corrected_enriched.txt`; un fichier `ocr_subtitles_timecoded_corrected.txt` produit `ocr_subtitles_timecoded_corrected_enriched.txt`.

## Step 19 - Video Summary

Produire un resume Markdown depuis le fichier enrichi, sous forme de tableau timecode/fait associe:

```powershell
python scripts/init/19_generate_video_summary.py
```

Le script lit `outputs/transcripts/*_enriched.txt` et ecrit `outputs/transcripts/video_summary.md`. Il conserve les lignes timecodees du fichier enrichi et les transforme en tableau Markdown avec une colonne `Timecode` et une colonne `Fait associe`.

La step suivante `20_create_plain_transcript.py` produit le fichier sans timecodes uniquement depuis `whisper_transcript_timecoded_corrected.txt`.

## Step 20 - Create Plain Transcript

Creer le transcript sans timecodes depuis la version corrigee:

```powershell
python scripts/init/20_create_plain_transcript.py
```

Le script lit uniquement `whisper_transcript_timecoded_corrected.txt` et produit `plain_transcript.txt`. Les fichiers `ocr_subtitles_timecoded_corrected.txt` ne generent pas de transcript plain.

## Step 21 - Create Transcript Chunks

Decouper les transcripts sans timecodes en chunks JSON, avec une limite de 1000 caracteres espaces compris et une coupe au prochain point apres depassement:

```powershell
python scripts/init/21_create_transcript_chunks.py
```

Le script lit d'abord `outputs/transcripts/plain_transcript.txt`, sinon `outputs/transcripts/ocr_subtitles.txt`, et ecrit `outputs/chunks/transcript_chunks.json`.

La detection des speakers combine les introductions du type `je m'appelle ...`, les noms propres detectes dans le transcript par spaCy, et les noms propres visibles dans `outputs/ocr/corrected_ocr_items.json` quand il existe, sinon `outputs/ocr/processed_ocr_items.json`, quand ils apparaissent dans un item OCR dont `kind` n'est ni `subtitle` ni `ocr_error`. Le modele transformer francais `fr_dep_news_trf` est charge sur GPU par defaut, et le filtre OCR evite de garder les prenoms simplement cites dans le transcript. Le modele et CuPy CUDA 12 sont declares dans `requirements.txt`; si l'environnement ne trouve pas le modele, le reinstaller avec:

```powershell
python -m spacy download fr_dep_news_trf
```

## Step 22 - Validate Chunk Speakers

Valider la liste des speakers detectes dans les chunks avec OpenAI:

```powershell
python scripts/init/22_validate_chunk_speakers.py
```

Le script lit `outputs/chunks/transcript_chunks.json`, recupere les valeurs `meta_data.speakers`, demande au modele quels speakers sont vraiment des personnes physiques, puis ecrit `outputs/chunks/transcript_chunks_speaker_validated.json`. Par defaut, le modele est `gpt-5.4-nano`, configurable avec `--model` ou `CHUNK_SPEAKER_VALIDATION_MODEL`; l'alias compact `gpt5.4nano` est aussi accepte.

Les chunks conservent leur contenu; seule la liste `speakers` est filtree. Une trace `speaker_validation` est ajoutee au JSON corrige avec les noms gardes et le nombre de rejets. Le script ecrit aussi `outputs/chunks/speaker_validation_log.json` avec la demande envoyee a GPT et sa reponse brute.

## Step 23 - Split Alert Chunks

Redecouper les chunks trop longs marques `ALERT`:

```powershell
python scripts/init/23_split_alert_chunks.py
```

Le script lit `outputs/chunks/transcript_chunks_speaker_validated.json` quand il existe, sinon `outputs/chunks/transcript_chunks.json`, et met a jour le fichier choisi en place.

## Step 24 - Create Transcript Embeddings

Creer les embeddings a partir des chunks:

```powershell
python scripts/init/24_create_chunk_embeddings.py
```

Le script lit `outputs/chunks/transcript_chunks_speaker_validated.json` quand il existe, sinon `outputs/chunks/transcript_chunks.json`, et ecrit un fichier JSON par chunk dans `outputs/chunks/`:

- `outputs/chunks/chunk_<index>_embedding.json`

## Step 25 - Upload Outputs To S3

Uploader le dernier dossier de videos vers le bucket S3 en conservant la meme arborescence:

```powershell
python scripts/init/25_upload_outputs_to_s3.py
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
python scripts/init/25_upload_outputs_to_s3.py --dry-run
python scripts/init/25_upload_outputs_to_s3.py --video-dir downloads/youtube/20260628_1312_init
python scripts/init/25_upload_outputs_to_s3.py --prefix youtube/20260628_1312_init
python scripts/init/25_upload_outputs_to_s3.py --clean-init-prefix
python scripts/init/25_upload_outputs_to_s3.py --force
```

## Step 26 - Update SQL Assets

Mettre a jour la base SQL avec le lien S3 du dossier video et synchroniser les transcripts disponibles:

```powershell
python scripts/init/26_update_sql_assets.py
```

Le script met a jour `videos` avec un seul lien S3 par dossier video via `s3_uri`. Il utilise le meme prefixe S3 `youtube/` que la Step 25 par defaut:

```text
downloads/youtube/20260628_1312_init/LJ-W6BjSJRo/LJ-W6BjSJRo.mp4
-> s3://bucket/youtube/20260628_1312_init/LJ-W6BjSJRo
```

Il met aussi a jour une seule ligne `transcripts` par video/langue avec les trois variantes trouvees dans chaque dossier `outputs/transcripts/`:

```text
transcript                   -> plain_transcript.txt
transcript_timecodes         -> whisper_transcript_timecoded_corrected.txt
transcript_timecodes_enrichi -> whisper_transcript_timecoded_corrected_enriched.txt
```

Options utiles:

```powershell
python scripts/init/26_update_sql_assets.py --dry-run
python scripts/init/26_update_sql_assets.py --video-dir downloads/youtube/20260628_1312_init
python scripts/init/26_update_sql_assets.py --prefix youtube/20260628_1312_init
python scripts/init/26_update_sql_assets.py --skip-transcripts
```

## Clear Database

Recreer la base SQL depuis le schema. Attention: cette commande supprime toutes les donnees des tables applicatives avant de recharger `docker/postgres/init/001_schema.sql`.

```powershell
python utils/reset_database.py
```

Vider les tables applicatives sans supprimer le schema:

```powershell
python utils/clear_database.py
```

## Outils locaux

Console HTML pour tester les requetes YouTube API:

```text
utils/api_console.html
```
