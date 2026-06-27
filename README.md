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
```

Les transcriptions utilisent une source non officielle et peuvent etre bloquees par YouTube. Ces variables permettent de ralentir les tentatives:

```powershell
YOUTUBE_API_SLEEP_SECONDS=2
TRANSCRIPT_RETRIES=2
TRANSCRIPT_RETRY_SECONDS=10
TRANSCRIPT_SLEEP_SECONDS=5
```

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

## Step 00 - Get Data

Importer les videos de la chaine, leurs statistiques du jour et leurs commentaires:

```powershell
python scripts/00_get_data.py
```

Le script travaille sur `https://www.youtube.com/@IONIS-STM/videos`, remplit les tables `videos`, `video_daily_stats`, `video_transcripts` et `comments`, puis ecrit aussi un fichier de titres dans `data/youtube/YYYY-MM-DD_HHMMSS.txt`.

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
