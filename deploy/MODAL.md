# Taches GPU sur Modal

Avec `PIPELINE_EXECUTION_BACKEND=modal`, le VPS fait tourner tout le pipeline,
et seules les trois taches GPU partent sur Modal :

| Tache | Ce que le VPS envoie | Ce que Modal renvoie |
| --- | --- | --- |
| `frames.classify` | `outputs/images` (les frames) | `outputs/images` rangees par classe + manifest |
| `ocr.extract_raw` | `outputs/images` classees | `outputs/ocr/raw` |
| `transcript.whisper` | l'audio MP3 16 kHz mono, extrait sur le VPS | `outputs/transcripts_whisper` |

La video elle-meme ne quitte jamais le VPS. Les videos longues (plus de
10 min) sautent les frames : seul l'audio part, environ 14 Mo pour 30 min.

`pipeline/gpu_offload.py` remplace le handler de ces taches dans l'executeur :
il emballe les entrees, appelle la fonction Modal `rag-ionis-gpu/run_gpu_task`,
puis remet les sorties dans le dossier video. Le reste (yt-dlp, OpenAI, S3,
PostgreSQL) se passe sur le VPS.

## Aucun secret chez Modal

La fonction ne recoit que des fichiers :
- **image** : Modal la construit depuis `deploy/modal_app.py`, avec les versions
  de `requirements-gpu.txt`. Pas de registre prive, pas de runner GPU.
- **modeles** : WhisperX et Paddle se telechargent librement dans le volume
  `rag-ionis-models`. Le modele pyannote, en acces restreint, y est televerse
  une fois depuis un poste qui l'a deja (voir plus bas).
- **appel** : le VPS s'authentifie avec `MODAL_TOKEN_ID` et `MODAL_TOKEN_SECRET`.

## Mise en place

1. Televerser le modele pyannote (une fois, depuis le poste de dev) :

   ```bash
   modal volume create rag-ionis-models
   ```

   ```bash
   modal volume put rag-ionis-models models/huggingface/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee /pyannote-community-1
   ```

2. Deployer et verifier le GPU :

   ```bash
   modal deploy deploy/modal_app.py
   ```

   ```bash
   modal run deploy/modal_app.py
   ```

   Ensuite, le workflow `deploy-modal.yml` redeploie a chaque push sur master
   qui touche `pipeline/` ou `deploy/modal_app.py`. Il lui faut les secrets
   GitHub `MODAL_TOKEN_ID` et `MODAL_TOKEN_SECRET` (environnement
   `production-modal`).

3. VPS : voir la section Production ci-dessous.

## Mesurer

`utils/modal_benchmark/run_benchmark.py` traite des videos archivees depuis
zero, avec les taches GPU sur Modal ou sur le GPU local, sans toucher a S3 ni a
PostgreSQL, et note la duree de chaque tache GPU :

```bash
python utils/modal_benchmark/run_benchmark.py --backend modal L0ipiwb9jS8 U-1vz4l4908 OPq7xoBFAYM
```

## Production

En service depuis le 8 novembre 2026, a la place de l'ancienne VM GPU
Scaleway. Sur le VPS, `/srv/rag_ionis/.env.production` contient
`PIPELINE_EXECUTION_BACKEND=modal`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` et
`YT_PROXY=socks5://host.docker.internal:1080`.

YouTube bloque l'IP du VPS : yt-dlp passe par le tunnel SSH `yt-tunnel` de
l'hote (projet agent_mots_cles). Le service compose `yt-proxy` le republie sur
la passerelle Docker 172.17.0.1, et une regle `ufw` laisse le reseau de
l'updater le joindre :

```bash
sudo ufw allow in from 172.18.0.0/16 to 172.17.0.1 port 1080 proto tcp
```

Lancer une daily sync a la main et suivre les lignes `[modal]` :

```bash
cd /srv/rag_ionis && docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production --profile jobs run --rm updater
```
