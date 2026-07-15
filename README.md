# RAG IONIS

Base locale pour construire un RAG.

## Interface RAG

Le repo contient maintenant une interface HTML/CSS branchee a un backend HTTP minimal.

Installer les dependances web si besoin:

```powershell
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn[standard]
```

Lancer le serveur:

```powershell
.\.venv\Scripts\python.exe -m uvicorn interface.app:app --host 127.0.0.1 --port 8000
```

Ouvrir ensuite:

- `http://127.0.0.1:8000/` pour l'interface
- `http://127.0.0.1:8000/health` pour verifier que l'API repond

Le endpoint `POST /api/rag`:

- embed la question avec OpenAI si une cle `OPENAI_API_KEY` est presente
- cherche les chunks en base via `pgvector`
- retombe sur une recherche SQL plein texte si besoin
- renvoie la reponse finale et les sources retenues

Le backend est decoupe par responsabilite:

- `interface/app.py`: creation FastAPI, cycle de vie et fichiers statiques;
- `interface/backend/api.py`: routes HTTP et conversion des erreurs en reponses API;
- `interface/backend/schemas.py` et `interface/backend/config.py`: contrats Pydantic et configuration;
- `interface/backend/database.py`: schema de chat, memoire et persistance SQL;
- `interface/backend/planner.py`: reformulation, planification et resolution des filtres;
- `interface/backend/retrieval.py`: SQL, BM25, recherche vectorielle, RRF et reranking;
- `interface/backend/generation.py`: evaluation des sources et generation de la reponse;
- `interface/backend/orchestration.py`: enchainement des etapes du RAG;
- `interface/backend/telemetry.py`: instrumentation Phoenix/OpenTelemetry.

## Observabilite RAG avec Phoenix

Le backend envoie des traces OpenTelemetry vers une instance locale d'Arize Phoenix.
Phoenix affiche la chronologie d'une requete RAG et le detail des etapes suivantes:

- reformulation et planner OpenAI;
- resolution des speakers;
- prefiltre SQL, BM25 et recherche vectorielle;
- fusion RRF et reranking Cohere;
- evaluation des sources et generation finale;
- ecriture du message dans PostgreSQL.

Lancer PostgreSQL et Phoenix:

```powershell
docker compose up -d postgres phoenix
```

L'interface Phoenix est disponible sur `http://127.0.0.1:6006/`. Elle n'est exposee
que sur la machine locale, car les traces peuvent contenir les questions, prompts,
reponses et chunks recuperes.

Le projet utilise un environnement Python unique pour le pipeline GPU, l'API et
Phoenix:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`start_app.bat` demarre automatiquement tous les services avec ce venv. Les variables utiles sont:

```text
PHOENIX_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_PROJECT_NAME=rag-ionis
```

Pendant la phase de validation, les traces techniques existantes restent stockees
dans `chat.messages`. La colonne `trace_id` relie chaque message a sa trace Phoenix.
Une fois la parite verifiee sur des requetes reelles, les colonnes JSONB techniques
pourront etre retirees progressivement sans toucher aux messages, reponses et sources.

## Environnement Python

Le seul venv est `.venv`. Il contient PyTorch CUDA, PaddleOCR, WhisperX, l'API,
OpenTelemetry/Phoenix et le pipeline de préparation des vidéos.

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
python -m pip install -r requirements.txt
```

Les index CUDA 12.6 de PyTorch et Paddle sont configurés directement dans
`requirements.txt`.

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
YTDLP_FORMAT=bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best
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

Migrer uniquement les embeddings existants vers `text-embedding-3-large` en 2000 dimensions, sans reconstruire les autres tables:

```powershell
.\.venv\Scripts\python.exe utils/migrate_embeddings_2000.py --dry-run
.\.venv\Scripts\python.exe utils/migrate_embeddings_2000.py
```

La migration est reprenable, conserve temporairement les anciens vecteurs dans `data.chunks.embedding_3072_backup` et cree l'index `idx_chunks_embedding_hnsw`. Apres validation de l'application, la sauvegarde peut etre supprimee manuellement:

```sql
ALTER TABLE data.chunks DROP COLUMN embedding_3072_backup;
```

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

Au demarrage, le script demande quels blocs executer: telechargement, traitement, upload S3 et mise a jour SQL. Si le telechargement est choisi et que des videos sont deja presentes dans `downloads/youtube/init`, il demande s'il faut les retelecharger. La reponse par defaut est non: les fichiers et sorties locales restent dans ce dossier unique. Repondre oui vide les anciens telechargements avant la Step 02.

Lorsque le retelechargement est demande, les anciens dossiers locaux sont remplaces:

- localement, toutes les videos et leurs sorties restent dans l'unique dossier `downloads/youtube/init`;
- l'upload S3 remplace automatiquement le prefixe `youtube/init` et supprime les anciens prefixes dates;
- la mise a jour SQL supprime et recree automatiquement le schema `data` avant de recharger les donnees.

Options utiles:

```powershell
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py 3 --skip-upload
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py all --skip-data
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py 3 --force
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py 3 --reuse-existing
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py 3 --redownload-existing
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py 3 --stages download,process
.\.venv\Scripts\python.exe scripts/init/RUN_PIPELINE_INIT.py all --stages s3,sql
```

`--force` regenere les sorties des etapes de traitement. Les options `--reuse-existing`
et `--redownload-existing` pilotent separement le telechargement et permettent une
execution non interactive.

Sans `--stages`, un menu demande de choisir un seul bloc avec `1`, `2`, `3` ou `4`,
puis affiche uniquement ses options. Avec `--stages`, les valeurs disponibles sont
`download`, `process`, `s3`, `sql` et `all`, et plusieurs blocs peuvent etre enchaines.
Chaque bloc propose ensuite ses options utiles:

- telechargement: actualisation automatique des metadonnees, reutilisation locale et cookies navigateur;
- traitement: regeneration, mode OpenAI, scope de review, modele de verification des images, correction et modele speakers;
- S3: purge automatique des anciens uploads puis envoi reel;
- SQL: recreation automatique du schema `data`, puis synchronisation complete des metadonnees, transcripts et chunks.

En mode `--stages`, les options omises utilisent les defaults non interactifs. Les
principaux overrides sont `--force` et `--image-review-model`.

## Step 02 - Download Videos

Telecharger en 360p les videos referencees dans la table SQL `videos`:

```powershell
python scripts/init/02_download_videos.py
```

Tous les lancements utilisent le meme dossier `downloads/youtube/init`, avec un sous-dossier par video. Sans `--force`, une video deja disponible localement est reutilisee sans nouvel appel a YouTube. Les anciens dossiers dates `*_init` sont automatiquement fusionnes puis supprimes:

```text
downloads/youtube/init/
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
Le fichier `metadata/youtube_video_metadata.json` conserve l'objet complet renvoye par
l'API YouTube pour la video, notamment `snippet.thumbnails` avec toutes les tailles
disponibles, `contentDetails`, `statistics`, `kind` et `etag`. Des champs plats derives
restent presents pour la compatibilite avec le pipeline.

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

La detection des speakers combine les introductions du type `je m'appelle ...`, les noms propres detectes dans le transcript par spaCy, et les noms propres visibles dans `outputs/ocr/corrected_ocr_items.json` quand il existe, sinon `outputs/ocr/processed_ocr_items.json`, quand ils apparaissent dans un item OCR dont `kind` n'est ni `subtitle` ni `ocr_error`. Le modele francais `fr_core_news_lg` fournit le parser et la reconnaissance d'entites sans imposer l'ancienne version de Protobuf incompatible avec OpenTelemetry. Le filtre OCR evite de garder les prenoms simplement cites dans le transcript. Le modele est declare dans `requirements.txt`; si l'environnement ne le trouve pas, le reinstaller avec:

```powershell
python -m spacy download fr_core_news_lg
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

Les embeddings de production utilisent `text-embedding-3-large` en 2000 dimensions. Un fichier existant en 3072 dimensions est automatiquement regenere. PostgreSQL les stocke dans `vector(2000)` avec un index HNSW cosine.

Une base deja peuplee en 3072 dimensions doit etre reconstruite apres regeneration des fichiers. Le pipeline complet le fait via l'etape SQL `--reset-database`; le script SQL refuse volontairement de tronquer silencieusement les anciens vecteurs.

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

Par defaut, le script prend `downloads/youtube/init` et l'upload dans le prefixe S3 `youtube/init`:

```text
downloads/youtube/init/LJ-W6BjSJRo/LJ-W6BjSJRo.mp4
-> s3://bucket/youtube/init/LJ-W6BjSJRo/LJ-W6BjSJRo.mp4
```

Options utiles:

```powershell
python scripts/init/25_upload_outputs_to_s3.py --dry-run
python scripts/init/25_upload_outputs_to_s3.py --video-dir downloads/youtube/init
python scripts/init/25_upload_outputs_to_s3.py --prefix youtube/init
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
downloads/youtube/init/LJ-W6BjSJRo/LJ-W6BjSJRo.mp4
-> s3://bucket/youtube/init/LJ-W6BjSJRo
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
python scripts/init/26_update_sql_assets.py --video-dir downloads/youtube/init
python scripts/init/26_update_sql_assets.py --prefix youtube/init
python scripts/init/26_update_sql_assets.py --skip-transcripts
python scripts/init/26_update_sql_assets.py --reset-database
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
utils/api_console/index.html
```
