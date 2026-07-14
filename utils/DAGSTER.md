# Dagster dans RAG IONIS

Dagster est l'orchestrateur unique du pipeline d'initialisation. Il exécute les scripts
métiers existants dans le venv GPU `.venv`, conserve l'historique des runs et présente
la chaîne de données sous forme d'assets partitionnés par identifiant YouTube.

## Démarrage

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start_app.bat
```

Ouvrir <http://127.0.0.1:3000>. `start_app.bat` est l'unique lanceur du projet : il
démarre PostgreSQL, Phoenix, les interfaces applicatives et Dagster. Il synchronise
également les vidéos locales comme partitions avant de démarrer Dagster.

## Importer de nouvelles vidéos

Dans **Jobs > importer_videos > Launchpad**, modifier les ressources si nécessaire:

```yaml
resources:
  import_settings:
    config:
      videos: "3"                 # entier, all, ou URL YouTube
      cookies_from_browser: edge
      force_download: false
      min_delay_seconds: 15
      max_delay_seconds: 45
      reset_before_import: true
```

`reset_before_import` est `true` par défaut. Il supprime les anciens téléchargements et
vide la base SQL avant l'import.

## Traiter une vidéo

Dans **Jobs > traiter_video**, cliquer **Materialize**, puis choisir la partition vidéo.
Le job `traiter_video` est configuré avec une concurrence maximale de 1, y compris
lorsqu'un backfill sélectionne plusieurs partitions vidéo : les runs `traiter_video`
attendent leur tour dans la file Dagster.
Le graphe exécute:

```text
video_source
  -> Steps 03..09
  -> Steps 10..14
  -> branche has_sub (15..23) OU no_sub (16..24)
  -> pipeline_outputs
```

La route est lue dans `metadata/pipeline_analysis.json`. Les assets de la branche qui
ne correspond pas à la vidéo sont matérialisés avec le statut `non applicable` et
n'exécutent aucun script.

Pour relancer seulement une partie, ouvrir **Catalog**, sélectionner l'asset désiré et
cliquer **Materialize selected**. Dagster permet aussi de relancer un run échoué depuis
le point de panne.

Les options du traitement sont modifiables dans le Launchpad:

```yaml
resources:
  pipeline_settings:
    config:
      force: false
      openai_mode: normal
      review_scope: duo
      correction_mode: balanced
      chunk_speaker_validation_model: gpt-5.4-nano
      image_review_model: gpt-5.6-luna
      dry_run_upload: false
      dry_run_sql: false
```

## Publier

- `publier_video` exécute seulement l'upload S3 et la mise à jour SQL;
- `pipeline_video_complet` enchaîne traitement et publication.

Les assets publient leurs durées, commandes, compteurs de fichiers et un lien vers
l'explorateur vidéo: <http://127.0.0.1:8001/videos>.

## Contrôles qualité

- fichier vidéo présent et non vide;
- au moins une image extraite;
- OCR non vide;
- transcript d'au moins 100 caractères;
- au moins un chunk;
- exactement un embedding par chunk.

Le code de l'orchestration se trouve dans `dagster_pipeline/` et l'historique local
dans `.dagster/` (ignoré par Git).
