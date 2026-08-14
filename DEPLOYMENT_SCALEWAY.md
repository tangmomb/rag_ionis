# Déployer l'interface RAG sur Scaleway

Cette configuration héberge l'interface RAG et PostgreSQL sur une Instance
Scaleway. Le pipeline vidéo peut continuer à tourner sur la machine GPU locale.
Les artefacts restent stockés dans le bucket Amazon S3 existant.

## Architecture

```text
Internet -> Caddy (HTTPS + Basic Auth) -> FastAPI -> PostgreSQL/pgvector
Machine GPU locale -------------------------------> PostgreSQL via tunnel SSH
Machine GPU locale -------------------------------> Amazon S3
```

PostgreSQL écoute uniquement sur la boucle locale du VPS. FastAPI n'est pas
publiée directement : Caddy est le seul service accessible depuis Internet.

## 1. Préparer Scaleway et le DNS

Créer une Instance Ubuntu récente avec au minimum 4 vCPU, 8 Go de RAM et un
volume dimensionné pour la base. Dans le groupe de sécurité, autoriser :

- TCP 22 depuis les adresses d'administration ;
- TCP 80 depuis Internet ;
- TCP et UDP 443 depuis Internet.

Ne pas ouvrir les ports 5432, 6006, 4317 ou 8006.

Créer ensuite un enregistrement DNS `A` pointant le domaine choisi vers l'IPv4
de l'Instance. Un enregistrement `AAAA` ne doit être ajouté que si l'IPv6 est
réellement configurée.

## 2. Installer Docker et récupérer le code

Installer Docker Engine et le plugin Docker Compose depuis le dépôt officiel
Docker, puis cloner le dépôt :

```bash
git clone https://github.com/tangmomb/rag_ionis.git
cd rag_ionis
git switch codex/deployment-scaleway
```

La branche de déploiement doit avoir été poussée avant d'exécuter ces commandes
sur le serveur.

## 3. Configurer les secrets

Créer la configuration locale du serveur :

```bash
cp .env.production.example .env.production
chmod 600 .env.production
```

Générer des mots de passe PostgreSQL aléatoires et URL-safe. Générer ensuite le
hash du mot de passe protégeant le site :

```bash
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'mot-de-passe-du-site'
```

Copier le résultat dans `APP_BASIC_AUTH_HASH` en conservant les apostrophes
autour du hash. Renseigner aussi :

- `APP_DOMAIN` ;
- `MISTRAL_API_KEY` ;
- `OPENAI_API_KEY` ;
- `COHERE_API_KEY` ;
- les deux mots de passe PostgreSQL.

Les clés Amazon S3 ne sont nécessaires que sur la machine qui exécute le
pipeline. Elles ne sont pas injectées dans le conteneur de l'API.

## 4. Valider et démarrer

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml config --quiet
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
docker compose --env-file .env.production -f docker-compose.prod.yml ps
```

Après la propagation DNS, ouvrir `https://<APP_DOMAIN>/health`. Le navigateur
demandera l'identifiant et le mot de passe configurés pour Basic Auth.

Pour consulter les journaux :

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml logs -f api caddy
```

## 5. Migrer la base locale

Créer le dump depuis le PostgreSQL local sans faire transiter le mot de passe
dans le dépôt :

```powershell
docker exec rag_ionis_postgres pg_dump -U rag_ionis -d rag_ionis -Fc -f /tmp/rag_ionis.dump
docker cp rag_ionis_postgres:/tmp/rag_ionis.dump .\rag_ionis.dump
scp .\rag_ionis.dump utilisateur@serveur:/tmp/rag_ionis.dump
```

Sur le VPS :

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml stop api
docker compose --env-file .env.production -f docker-compose.prod.yml cp /tmp/rag_ionis.dump postgres:/tmp/rag_ionis.dump
docker compose --env-file .env.production -f docker-compose.prod.yml exec postgres sh -lc 'pg_restore --clean --if-exists --no-owner --exit-on-error -U "$POSTGRES_USER" -d "$POSTGRES_DB" /tmp/rag_ionis.dump'
docker compose --env-file .env.production -f docker-compose.prod.yml start api
```

Le retour non nul de `pg_restore` doit être examiné avant de considérer la
migration terminée. Vérifier ensuite `/health` et effectuer une question test.

## 6. Publier de nouvelles données depuis la machine GPU

Le port PostgreSQL du VPS est lié à `127.0.0.1`. Ouvrir un tunnel depuis la
machine locale :

```powershell
ssh -N -L 55432:127.0.0.1:5432 utilisateur@serveur
```

Dans un autre terminal, utiliser temporairement une URL de publication qui
pointe vers le tunnel :

```powershell
$env:DATABASE_URL='postgresql://rag_ionis:MOT_DE_PASSE@127.0.0.1:55432/rag_ionis'
.\.venv\Scripts\python.exe -m pipeline.publish.upload_outputs_to_s3
.\.venv\Scripts\python.exe -m pipeline.publish.sync_database
```

La première commande continue d'utiliser Amazon S3 avec les variables
`S3_BUCKET_NAME`, `S3_REGION`, `S3_ACCESS_KEY_ID` et `S3_SECRET_ACCESS_KEY`.

## 7. Phoenix optionnel

Phoenix n'est pas lancé par défaut. Pour l'activer :

```bash
PHOENIX_ENABLED=true docker compose --env-file .env.production -f docker-compose.prod.yml --profile observability up -d
```

Son interface reste liée à `127.0.0.1:6006`. Pour la consulter :

```bash
ssh -N -L 6006:127.0.0.1:6006 utilisateur@serveur
```

Puis ouvrir `http://127.0.0.1:6006` localement.

## 8. Sauvegardes et mises à jour

Créer régulièrement un dump hors du volume Docker :

```bash
mkdir -p backups
docker compose --env-file .env.production -f docker-compose.prod.yml exec -T postgres sh -lc 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "backups/rag_ionis_$(date +%Y%m%d_%H%M%S).dump"
```

Copier ces sauvegardes sur un autre stockage. Un volume Docker ou un snapshot de
l'Instance ne remplace pas une sauvegarde indépendante testée.

Pour mettre l'application à jour :

```bash
git pull --ff-only
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
docker compose --env-file .env.production -f docker-compose.prod.yml ps
```
