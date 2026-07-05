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

Sans argument, le script demande quoi traiter: `all` pour toute la chaine, un nombre pour tester, ou un lien YouTube precis.

Tester le pipeline sur un nombre limite de videos:

```powershell
.\.venv\Scripts\python.exe scripts/init/run_pipeline_init.py 3
```

Tester le pipeline sur une video precise:

```powershell
.\.venv\Scripts\python.exe scripts/init/run_pipeline_init.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Le script lance les steps 00 a 16. Au demarrage, il vide les tables applicatives SQL en conservant le schema, puis supprime les anciens dossiers locaux `*_init` dans `downloads/youtube/`. La Step 02 cree ensuite un nouveau dossier date suffixe `_init`, puis ce meme dossier est passe aux steps suivantes.

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
.\.venv\Scripts\python.exe scripts/init/run_pipeline_init.py 3 --image-clusters 2
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

## Step 03 - Extract Images

Extraire une image toutes les 2 secondes pour chaque video:

```powershell
python scripts/init/03_extract_images.py
```

Le script cree un dossier `images/` dans chaque dossier video. Les images sont nommees par timecode minute/seconde, par exemple `00_00.jpg`, `00_02.jpg`, `01_00.jpg`.

## Step 04 - Classify Images

Classer localement les images extraites avec des features OpenCV simples et k-means:

```powershell
python scripts/init/04_classify_images.py
```

Le script lit `images/`, applique un flou pour limiter l'impact du texte, calcule des features simples (`gray_std`, `color_std`, `edge_ratio`, couleur dominante, luminosite, regions plates), lance k-means en 2 clusters uniquement si une sequence d'images contient un aplat ou un degrade stable, puis reorganise directement les fichiers dans deux dossiers finaux. Si le split est net, le plus gros cluster va dans `images/answers/` et l'autre va dans `images/graphic/`. Dans `graphic/`, les images qui se suivent numeriquement sont regroupees en sous-dossiers `graphic_01/`, `graphic_02/`, etc.

Pour les videos sans vrai chapitrage graphique, le script ne force pas de faux cluster: si aucune sequence d'images avec aplat/degrade n'est detectee, si la separation est trop faible, si le plus gros cluster contient moins de 70% des images, ou si le petit cluster contient moins de 5 images, toutes les images vont dans `images/no_cluster/`. Le manifeste indique alors `cluster_identifiable: false` avec les raisons dans `no_graphic_reasons` et `kmeans_run: false` quand le preflight a bloque le clustering.

Il ecrit aussi `images/manifest.json` avec le role de chaque image et `images/cv_features.json` avec les mesures OpenCV. La Step 05 OCR lit ensuite les images recursivement et conserve ces chemins relatifs dans ses JSON.

Par defaut, la sortie finale est binaire: `answers` ou `graphic`.
Quand un ecran graphique contient un portrait integre, k-means peut le rapprocher des reponses. Un override rattache alors l'image a `graphic` si le fond garde une couleur tres dominante (`--graphic-dominant-hue-ratio 0.70`) et tres peu de contours apres flou (`--graphic-max-edge-ratio 0.002`).

Options utiles:

```powershell
python scripts/init/04_classify_images.py --clusters 2
python scripts/init/04_classify_images.py --blur-kernel 31
python scripts/init/04_classify_images.py --feature-size 64
python scripts/init/04_classify_images.py --min-cluster-images 5
python scripts/init/04_classify_images.py --min-majority-ratio 0.70
python scripts/init/04_classify_images.py --min-silhouette 0.12
python scripts/init/04_classify_images.py --graphic-dominant-hue-ratio 0.70
python scripts/init/04_classify_images.py --graphic-max-edge-ratio 0.002
python scripts/init/04_classify_images.py --min-flat-region-ratio 0.08
python scripts/init/04_classify_images.py --min-flat-component-ratio 0.03
python scripts/init/04_classify_images.py --min-flat-images 2
python scripts/init/04_classify_images.py --force
```

## Step 05 - numero_ocr_brut

Extraire localement l'OCR brut avec PaddleOCR sur toutes les images:

```powershell
python scripts/init/05_numero_ocr_brut.py
```

Le script lit toutes les images dans `images/` et ecrit `transcript/<video_id>_ocr_brut.json`. Il ne produit plus directement le `processed.json`.

## Step 06 - Build OCR Processed

Transformer l'OCR brut en OCR traite sans relancer PaddleOCR:

```powershell
python scripts/init/05b_images_ocr_postprocess.py
```

Le script lit `transcript/<video_id>_ocr_brut.json` et ecrit `transcript/<video_id>_ocr_processed.json`. Par defaut, seules les detections OCR avec `rec_score >= 0.9` sont conservees dans le JSON traite, puis les textes de decor probables sont filtres par taille, isolement, persistance statique avec variantes OCR proches, fragments progressifs et liste d'exclusion legere. Les detections provenant de `images/graphic/graphic_XX/` sont conservees avec `kind: "graphic_XX"`; le dernier dossier graphique est conserve avec `kind: "outro"`; pour chaque dossier `graphic_XX`, le JSON traite ne garde que la frame qui produit le plus de texte OCR. Pour les detections non sous-titres venant de `images/answers/`, un meme mot ou une meme phrase repete dans une fenetre de 20 frames ne garde que sa derniere occurrence. Les textes non sous-titres finissant par `?` sont classes comme `question_intertitle`. Les sous-titres OCR sont detectes par une ligne de position relative recurrente, principalement le centre X commun des boites, avec des garde-fous geometriques sur Y, largeur et hauteur.

## Step 07 - OCR Subtitles

Concatener les items OCR de type `subtitle` dans un fichier texte dedie, avec une version timecodee en parallele:

```powershell
python scripts/init/06_ocr_subtitles.py
```

Le script lit `transcript/<video_id>_ocr_processed_corrected.json` quand il existe, sinon `transcript/<video_id>_ocr_processed.json`, et ecrit `transcript/<video_id>_ocr_subtitle.txt` ainsi que `transcript/<video_id>_ocr_subtitle_timecodes.txt`. Dans le pipeline init, il s'appuie directement sur `transcript/<video_id>_ocr_processed.json`.

## Step 08 - Whisper Transcription

Transcrire localement avec `whisperx` les videos du dernier dossier de telechargement, sauf si des sous-titres OCR sont deja presents:

```powershell
python scripts/init/07_whisper_transcription.py
```

Le script extrait un fichier audio temporaire avec ffmpeg, transcrit localement avec `whisperx` en francais sur GPU, puis aligne les segments pour produire des timecodes. Il cree un dossier `transcript/` dans chaque dossier video et produit un fichier horodate, par exemple `hGUkhjssd_transcript_timecodes.txt`.

Les parametres utiles se reglant via `.env` sont `WHISPERX_MODEL`, `WHISPERX_LANGUAGE`, `WHISPERX_DEVICE`, `WHISPERX_COMPUTE_TYPE` et `WHISPERX_BATCH_SIZE`.

Pour forcer un usage GPU, garde `WHISPERX_DEVICE=cuda` et `WHISPERX_COMPUTE_TYPE=float16`. Pour des machines plus legeres, `WHISPERX_DEVICE=cpu` et `WHISPERX_COMPUTE_TYPE=int8` restent possibles. Le modele par defaut est maintenant `large-v3`.

Quand la transcription produit des timecodes, le fichier se termine par `_transcript_timecodes.txt`.

## Step 09 - Correct Timecodes

Corriger certains mots du transcript timecode en les comparant aux mots OCR trouves dans `processed.json`, pour essayer de recuperer des noms propres visibles a l'ecran:

```powershell
python scripts/init/08_correct_timecodes.py
```

Le script lit soit `transcript/*_transcript_timecodes.txt`, soit `transcript/*_ocr_subtitle_timecodes.txt` quand le premier n'existe pas, puis ecrit un nouveau fichier avec `_corrected.txt` a la fin. Il conserve la casse reelle vue par l'OCR et se concentre sur les zones `name`, `lower_third`, `title`, `logo` et `graphic_XX` pour limiter les faux positifs.

Le niveau de correction est ajustable:

```powershell
python scripts/init/08_correct_timecodes.py --mode conservative
python scripts/init/08_correct_timecodes.py --mode balanced
python scripts/init/08_correct_timecodes.py --mode aggressive
```

`conservative` corrige peu, `aggressive` accepte plus de noms proches, et `balanced` est le defaut.

## Step 10 - Enrich Timecodes

Ajouter les textes visibles a l'ecran dans les timecodes corriges:

```powershell
python scripts/init/09_enrich_transcripts.py
```

Le script n'appelle aucune API. Il combine les fichiers corriges avec `transcript/*_ocr_processed_corrected.json` quand il existe, sinon `transcript/*_ocr_processed.json`, et ajoute seulement le suffixe `_enrichi.txt` au fichier source. Un fichier `*_transcript_timecodes_corrected.txt` produit donc `*_transcript_timecodes_corrected_enrichi.txt`; un fichier `*_ocr_subtitle_timecodes_corrected.txt` produit `*_ocr_subtitle_timecodes_corrected_enrichi.txt`, sans creer de faux fichier `transcript`.

## Step 11 - Video Summary

Produire un resume Markdown depuis le fichier enrichi, sous forme de tableau timecode/fait associe:

```powershell
python scripts/init/09_generate_video_summary.py
```

Le script lit `transcript/*_enrichi.txt` et ecrit `transcript/*_video_summary.md`. Il conserve les lignes timecodees du fichier enrichi et les transforme en tableau Markdown avec une colonne `Timecode` et une colonne `Fait associé`.

La step suivante `10_strip_timecodes.py` produit le fichier sans timecodes uniquement depuis `*_transcript_timecodes_corrected.txt`.

## Step 12 - Strip Timecodes

Creer le transcript sans timecodes depuis la version corrigee:

```powershell
python scripts/init/10_strip_timecodes.py
```

Le script lit uniquement `*_transcript_timecodes_corrected.txt` et produit `*_transcript.txt`. Les fichiers `*_ocr_subtitle_timecodes_corrected.txt` ne generent pas de transcript plain.

## Step 13 - Create Transcript Chunks

Decouper les transcripts sans timecodes en chunks JSON, avec une limite de 1000 caracteres espaces compris et une coupe au prochain point apres depassement:

```powershell
python scripts/init/11_create_chunks.py
```

Le script lit d'abord `transcript/<video_id>_transcript.txt`, sinon `transcript/<video_id>_ocr_subtitle.txt`, et ecrit `chunks/<video_id>_chunks.json`.

La detection des speakers combine les introductions du type `je m'appelle ...`, les noms propres detectes dans le transcript par spaCy, et les noms propres visibles dans `transcript/<video_id>_ocr_processed_corrected.json` quand il existe, sinon `transcript/<video_id>_ocr_processed.json`, quand ils apparaissent dans un item OCR dont `kind` n'est ni `subtitle` ni `ocr_error`. Le modele transformer francais `fr_dep_news_trf` est charge sur GPU par defaut, et le filtre OCR evite de garder les prenoms simplement cites dans le transcript. Le modele et CuPy CUDA 12 sont declares dans `requirements.txt`; si l'environnement ne trouve pas le modele, le reinstaller avec:

```powershell
python -m spacy download fr_dep_news_trf
```

## Step 14 - Validate Chunk Speakers

Valider la liste des speakers detectes dans les chunks avec OpenAI:

```powershell
python scripts/init/12_validate_chunk_speakers.py
```

Le script lit `chunks/<video_id>_chunks.json`, recupere les valeurs `meta_data.speakers`, demande au modele quels speakers sont vraiment des personnes physiques, puis ecrit `chunks/<video_id>_chunks_corrected.json`. Par defaut, le modele est `gpt-5.4-nano`, configurable avec `--model` ou `CHUNK_SPEAKER_VALIDATION_MODEL`; l'alias compact `gpt5.4nano` est aussi accepte.

Les chunks conservent leur contenu; seule la liste `speakers` est filtree. Une trace `speaker_validation` est ajoutee au JSON corrige avec les noms gardes et le nombre de rejets. Le script ecrit aussi `chunks/<video_id>_chunk_speaker_validation_log.json` avec la demande envoyee a GPT et sa reponse brute.

## Step 15 - Split Alert Chunks

Redecouper les chunks trop longs marques `ALERT`:

```powershell
python scripts/init/12_split_alert_chunks.py
```

Le script lit `chunks/*_chunks_corrected.json` quand il existe, sinon `chunks/*_chunks.json`, et met a jour le fichier choisi en place.

## Step 16 - Create Transcript Embeddings

Creer les embeddings a partir des chunks:

```powershell
python scripts/init/13_create_embeddings.py
```

Le script lit `chunks/<video_id>_chunks_corrected.json` quand il existe, sinon `chunks/<video_id>_chunks.json`, et ecrit un fichier JSON par chunk dans `chunks/`:

- `chunks/<video_id>_chunk_<index>_embedding.json`

## Step 17 - Upload Videos To S3

Uploader le dernier dossier de videos vers le bucket S3 en conservant la meme arborescence:

```powershell
python scripts/init/14_upload_s3.py
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
python scripts/init/14_upload_s3.py --dry-run
python scripts/init/14_upload_s3.py --video-dir downloads/youtube/20260628_1312_init
python scripts/init/14_upload_s3.py --prefix youtube/20260628_1312_init
python scripts/init/14_upload_s3.py --clean-init-prefix
python scripts/init/14_upload_s3.py --force
```

## Step 18 - Update SQL Assets

Mettre a jour la base SQL avec les chemins S3 des fichiers generes et synchroniser les transcripts disponibles:

```powershell
python scripts/init/15_upload_sql.py
```

Le script cree la table `video_elements` si elle n'existe pas, puis y enregistre les videos, images, analyses et transcripts du dernier dossier de `downloads/youtube/`. Il utilise le meme prefixe S3 `youtube/` que la Step 17 par defaut:

```text
downloads/youtube/20260628_1312_init/LJ-W6BjSJRo/LJ-W6BjSJRo.mp4
-> s3://bucket/youtube/20260628_1312_init/LJ-W6BjSJRo/LJ-W6BjSJRo.mp4
```

Il met aussi a jour une seule ligne `video_transcripts` par video/langue avec les trois variantes trouvees dans chaque dossier `transcript/`:

```text
transcript                  -> *_transcript.txt
transcript_timecodes        -> *_transcript_timecodes_corrected.txt
transcript_timecodes_enrichi -> *_transcript_timecodes_corrected_enrichi.txt
```

Options utiles:

```powershell
python scripts/init/15_upload_sql.py --dry-run
python scripts/init/15_upload_sql.py --video-dir downloads/youtube/20260628_1312_init
python scripts/init/15_upload_sql.py --prefix youtube/20260628_1312_init
python scripts/init/15_upload_sql.py --clean-init-assets
python scripts/init/15_upload_sql.py --skip-transcripts
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
