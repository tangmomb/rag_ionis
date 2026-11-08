# Worker GPU sur Modal (prototype)

`PIPELINE_EXECUTION_BACKEND=modal` remplace la VM GPU Scaleway par une fonction
Modal. Le VPS envoie le même `job_input` et récupère les artefacts sur le même
S3 ; seuls le démarrage, l'attente et l'arrêt changent.

| Scaleway | Modal |
| --- | --- |
| `ScalewayClient` allume la VM, attend `started`, l'éteint | `Function.spawn(job_input)` puis `FunctionCall.get()` |
| `job.json` / `status.json` dans S3, worker `--poll` | appel direct, la sortie de `process_job` revient au VPS |
| unité systemd, `docker pull`, connexion au registre | Modal tire l'image une fois et la met en cache |
| `/var/lib/rag-ionis/models` sur le disque de la VM | volume Modal `rag-ionis-models` |
| L4 parfois indisponible dans une zone | Modal choisit un L4 libre |

Le code exécuté est celui de l'image GPU publiée (`Dockerfile.scaleway`) :
Modal la réutilise telle quelle, il n'y a pas de seconde image à maintenir.

## Mise en place

1. Installer le client et s'authentifier sur le poste qui déploie :

   ```bash
   pip install modal
   modal token new
   ```

2. Créer les deux secrets. Le premier donne accès au registre Scaleway, le
   second reprend les variables de `/etc/rag-ionis/scaleway-worker.env`
   (S3, OpenAI, YouTube, Hugging Face) :

   ```bash
   modal secret create rag-ionis-registry REGISTRY_USERNAME=nologin REGISTRY_PASSWORD=<SCW_SECRET_KEY>
   ```

   ```bash
   modal secret create rag-ionis-worker --from-dotenv deploy/scaleway-worker.env
   ```

3. Vérifier le GPU (Torch, Paddle et WhisperX sur le L4), puis déployer :

   ```bash
   SCALEWAY_CONTAINER_IMAGE=rg.fr-par.scw.cloud/NAMESPACE/rag-ionis-scaleway:VERSION modal run deploy/modal_app.py
   ```

   ```bash
   SCALEWAY_CONTAINER_IMAGE=rg.fr-par.scw.cloud/NAMESPACE/rag-ionis-scaleway:VERSION modal deploy deploy/modal_app.py
   ```

4. Côté client (poste local ou VPS) : `pip install modal`, puis
   `PIPELINE_EXECUTION_BACKEND=modal` avec `MODAL_TOKEN_ID` et
   `MODAL_TOKEN_SECRET`. Exemple sur une vidéo déjà archivée :

   ```bash
   PIPELINE_EXECUTION_BACKEND=modal python -m pipeline run <video_id> --root downloads/youtube/<archive>
   ```

## Options de déploiement

- `MODAL_MAX_CONTAINERS` (défaut `1`) : nombre de vidéos traitées en parallèle.
  Le client envoie encore les vidéos une par une ; le paralléliser est l'étape
  suivante si le prototype est retenu.
- `MODAL_REGION` : par exemple `eu`. Le choix de région dépend de l'offre Modal
  et peut majorer le prix.

## Avant de remplacer Scaleway

- **Hébergement.** Les vidéos et les clés passent par Modal (société
  américaine), alors que la FAQ du site annonce un hébergement Scaleway.
- **yt-dlp.** La daily sync télécharge YouTube depuis le worker : vérifier que
  les IP de Modal ne sont pas bloquées.
- **Image.** Modal installe son client dans l'image. Le premier `modal run`
  confirme que ses dépendances ne cassent pas Torch ou Paddle.
- **VPS.** `modal` n'est pas encore dans le groupe `vps` de `pyproject.toml` :
  l'ajouter puis lancer `uv lock` avant de basculer la production.
- **Coût.** Modal facture les secondes de GPU utilisées. Comparer avec le prix
  horaire du L4 Scaleway, démarrage de la VM et stockage compris.
