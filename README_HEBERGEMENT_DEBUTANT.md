# Hébergement de RAG IONIS — guide débutant

Ce document explique avec des mots simples l'infrastructure du projet, ce qui a
été fait pour le mettre en ligne et les commandes utiles pour l'administrer.

L'application est accessible publiquement à cette adresse :

<https://ionis.tanguym.fr>

## 1. Ce qui est hébergé où

```text
Visiteur
   |
   | https://ionis.tanguym.fr
   v
DNS chez IONOS
   |
   | pointe vers 179.237.98.117
   v
Pare-feu Infomaniak (ports 80 et 443)
   |
   v
VPS Ubuntu Infomaniak
   |
   +-- Caddy : reçoit le trafic web et gère HTTPS
   +-- FastAPI : exécute l'interface et l'API du RAG
   +-- PostgreSQL + pgvector : stocke les données et les vecteurs

Machine locale avec GPU
   +-- exécute les traitements lourds du pipeline
   +-- envoie les fichiers dans Amazon S3
   +-- peut synchroniser la base du VPS par un tunnel SSH
```

Il n'est donc pas nécessaire d'avoir trois hébergements séparés. Le VPS fait
tourner le site, l'API et la base de données. Le GPU peut rester sur la machine
locale et Amazon S3 reste le stockage des fichiers.

## 2. Les notions importantes

### VPS

Un VPS est un ordinateur Linux distant, allumé en permanence. Le nôtre est chez
Infomaniak et utilise Ubuntu 24.04. Son IPv4 publique est `179.237.98.117`.

### Nom de domaine et DNS

Le domaine est géré chez IONOS. L'enregistrement DNS de type `A` fait
correspondre `ionis.tanguym.fr` à `179.237.98.117`.

Le DNS donne l'adresse du serveur, mais ne lance pas l'application. Docker et
Caddy doivent aussi fonctionner sur le VPS.

Nous avons supprimé l'ancien enregistrement `AAAA`, car il envoyait les
visiteurs en IPv6 vers une mauvaise destination. Il ne faudra en recréer un que
si l'IPv6 du VPS est réellement configurée.

### Pare-feu et ports

Un port est une porte réseau. Dans le pare-feu Infomaniak, nous avons ouvert :

- `22/TCP` pour administrer le VPS avec SSH ;
- `80/TCP` pour HTTP et la validation des certificats ;
- `443/TCP` pour HTTPS ;
- éventuellement `443/UDP` pour HTTP/3.

Les ports internes `5432`, `6006`, `4317` et `8006` ne doivent pas être ouverts
à Internet. La base PostgreSQL écoute seulement sur `127.0.0.1:5432`.

### SSH et la clé SSH

SSH sert à ouvrir un terminal sur le VPS depuis le PC :

```powershell
ssh -i C:\Users\rgb\.ssh\rag_ionis_infomaniak_ed25519 ubuntu@179.237.98.117
```

La clé SSH comprend deux parties :

- la clé privée reste uniquement sur le PC et ne doit jamais être partagée ;
- la clé publique a été enregistrée chez Infomaniak.

La _fingerprint_ est l'empreinte qui identifie la clé. La _randomart_ n'est
qu'une représentation visuelle de cette empreinte. Ce ne sont pas des mots de
passe pour le site.

La passphrase protège la clé privée sur le PC. Elle peut être différente de
tous les autres mots de passe du projet.

### Git et GitHub

Git conserve l'historique du code. GitHub héberge une copie du dépôt, que le VPS
peut télécharger.

La version déployée utilise une branche Git dédiée au déploiement sur le VPS
Infomaniak. La configuration repose sur Docker et Ubuntu.

Le dépôt a été rendu public pour simplifier le clonage. Cela ne doit jamais
rendre publics les secrets décrits plus bas.

### Docker et Docker Compose

Docker lance chaque partie de l'application dans un conteneur isolé. Le fichier
`docker-compose.prod.yml` décrit les services :

- `postgres` : PostgreSQL avec l'extension pgvector ;
- `api` : FastAPI et l'interface ;
- `caddy` : serveur web public et HTTPS ;
- `phoenix` : enregistre les conversations et les traces du RAG.

Docker Compose démarre et relie ces services avec une seule commande.

### Caddy et HTTPS

Caddy reçoit les connexions sur les ports 80 et 443, obtient automatiquement un
certificat HTTPS auprès de Let's Encrypt, puis transmet les requêtes au service
FastAPI sur le réseau privé Docker.

Au départ, Let's Encrypt échouait avec un délai d'attente : les ports 80 et 443
étaient fermés dans le pare-feu Infomaniak. Après leur ouverture, le certificat
a été généré et HTTPS a fonctionné.

### PostgreSQL et pgvector

PostgreSQL est la base de données. L'extension pgvector permet d'y conserver les
vecteurs utilisés pour la recherche sémantique du RAG.

Le volume Docker `postgres_data` conserve les données lorsque le conteneur est
redémarré ou recréé. Un volume ne remplace toutefois pas une sauvegarde.

### Amazon S3 et le GPU

Amazon S3 conserve les fichiers et artefacts produits par le pipeline. Les
identifiants S3 restent sur la machine qui exécute le pipeline et ne sont pas
nécessaires dans le conteneur de l'API actuelle.

Le traitement vidéo ou les calculs lourds peuvent continuer sur la machine GPU
locale. Le VPS n'a donc pas besoin de GPU pour servir l'interface et effectuer
les recherches déjà préparées.

### Phoenix

Phoenix démarre automatiquement avec les autres services. Chaque requête faite
à l'interface RAG y enregistre la question, la réponse et les principales étapes
internes, regroupées par conversation. Ces traces sont conservées dans le volume
Docker `phoenix_data`, même lorsque le conteneur est recréé.

Le port `6006` reste privé : l'interface Phoenix n'est pas exposée directement à
Internet. Elle se consulte depuis le PC avec le tunnel SSH expliqué plus bas.

## 3. Les différents secrets — à ne pas confondre

### 1. Clé et passphrase SSH

Elles servent uniquement à se connecter au terminal du VPS. Elles ne servent
pas à se connecter au site web.

### 2. Variables de `.env.production`

Le fichier `~/rag_ionis/.env.production` existe uniquement sur le VPS. Il
contient notamment :

- le domaine ;
- les mots de passe PostgreSQL ;
- les clés API Mistral, OpenAI et Cohere.

Ce fichier ne doit jamais être envoyé sur GitHub, collé dans une conversation
ou rendu public. Ses permissions ont été limitées avec `chmod 600`.

## 4. Ce qui a été fait pendant le déploiement

1. Le travail local a été sauvegardé dans Git avec un commit, puis poussé sur
   GitHub.
2. Une branche de déploiement a ajouté le Dockerfile de l'API, Docker Compose,
   Caddy et un exemple de configuration de production.
3. Un VPS Linux Ubuntu a été créé chez Infomaniak.
4. Une clé SSH a été générée sur le PC et sa partie publique a été donnée à
   Infomaniak.
5. Docker et Docker Compose ont été installés et testés sur le VPS.
6. Le dépôt GitHub et la branche de déploiement ont été clonés sur le VPS.
7. `.env.production` a été créé avec le domaine, les mots de passe de base et
   les clés API.
8. Chez IONOS, le DNS `A` de `ionis.tanguym.fr` a été dirigé vers
   `179.237.98.117` et l'ancien `AAAA` a été supprimé.
9. Les services PostgreSQL, FastAPI et Caddy ont été construits et démarrés.
10. Les ports 80 et 443 ont été ouverts dans le pare-feu Infomaniak.
11. Caddy a obtenu le certificat HTTPS et le site est devenu accessible
    publiquement.

## 5. Commandes quotidiennes sur le VPS

Commencer par se connecter en SSH, puis :

```bash
cd ~/rag_ionis
```

### Voir l'état des services

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml ps
```

`postgres`, `phoenix` et `api` doivent normalement être indiqués comme `healthy`,
et `caddy` comme démarré.

### Voir les journaux

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml logs --tail=100 api caddy postgres phoenix
```

Pour suivre les nouveaux messages en direct :

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml logs -f api caddy
```

Quitter le suivi avec `Ctrl+C`. Cela n'arrête pas les services.

### Consulter les conversations dans Phoenix

Depuis le PC, ouvrir un tunnel SSH et laisser ce terminal ouvert :

```powershell
ssh -i C:\Users\rgb\.ssh\rag_ionis_infomaniak_ed25519 -N -L 6006:127.0.0.1:6006 ubuntu@179.237.98.117
```

Ouvrir ensuite <http://127.0.0.1:6006> dans le navigateur. Les requêtes de
l'interface apparaissent dans le projet `rag-ionis` et sont regroupées par
conversation.

### Redémarrer les services

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml restart
```

### Arrêter puis relancer

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml down
docker compose --env-file .env.production -f docker-compose.prod.yml up -d
```

Ne pas ajouter `-v` à la commande `down` : cela demanderait aussi la suppression
des volumes et pourrait supprimer les données PostgreSQL.

### Vérifier le site

```bash
curl -I https://ionis.tanguym.fr
curl https://ionis.tanguym.fr/health
```

Une réponse `200` indique normalement que la ressource demandée fonctionne. Une
réponse `502` ou `503` signifie généralement que Caddy fonctionne, mais pas
l'API derrière lui.

## 6. Mettre l'application à jour

Après avoir modifié, testé, commité et poussé le code depuis le PC, se connecter
au VPS et exécuter :

```bash
cd ~/rag_ionis
git status
git pull --ff-only
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
docker compose --env-file .env.production -f docker-compose.prod.yml ps
```

`git status` permet de vérifier qu'aucune modification manuelle du VPS ne sera
écrasée. Les modifications de configuration effectuées directement sur le VPS
doivent être reportées dans Git ; sinon le prochain `git pull` pourra échouer ou
rétablir une ancienne configuration.

## 7. Sauvegarder la base de données

Créer une sauvegarde sur le VPS :

```bash
cd ~/rag_ionis
mkdir -p backups
docker compose --env-file .env.production -f docker-compose.prod.yml exec -T postgres sh -lc 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "backups/rag_ionis_$(date +%Y%m%d_%H%M%S).dump"
```

Il faut ensuite copier régulièrement les sauvegardes hors du VPS. Une sauvegarde
qui reste uniquement sur le même serveur disparaîtra avec lui en cas de panne
grave.

Les sauvegardes peuvent contenir des données sensibles : ne pas les mettre dans
Git.

## 8. Connecter temporairement la machine locale à PostgreSQL

Le port de la base n'est volontairement pas public. Pour y accéder depuis le PC,
ouvrir un tunnel SSH dans un terminal PowerShell et le laisser ouvert :

```powershell
ssh -i C:\Users\rgb\.ssh\rag_ionis_infomaniak_ed25519 -N -L 55432:127.0.0.1:5432 ubuntu@179.237.98.117
```

Les outils locaux peuvent alors joindre la base sur `127.0.0.1:55432`. Le mot de
passe utilisé est `POSTGRES_PASSWORD` dans `.env.production`, pas la passphrase
SSH.

## 9. Dépannage rapide

### Le domaine n'affiche rien

Vérifier successivement :

```powershell
nslookup ionis.tanguym.fr
Test-NetConnection ionis.tanguym.fr -Port 443
```

Puis sur le VPS :

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml ps
docker compose --env-file .env.production -f docker-compose.prod.yml logs --tail=100 caddy api
```

### Erreur de certificat ou TLS

Vérifier que le DNS pointe vers `179.237.98.117`, qu'aucun mauvais `AAAA`
n'existe et que les ports TCP 80 et 443 sont ouverts chez Infomaniak.

### Erreur 502 ou 503

Examiner l'état et les logs du service `api`, puis ceux de `postgres`.

### Une variable manque

Ouvrir le fichier sans afficher ses secrets dans le terminal :

```bash
nano .env.production
```

Enregistrer dans Nano avec `Ctrl+O`, valider avec `Entrée`, puis quitter avec
`Ctrl+X`.

## 10. Règles de sécurité à retenir

- Ne jamais publier `.env.production`.
- Ne jamais partager la clé SSH privée ni les clés API.
- Ne jamais ouvrir PostgreSQL (`5432`) directement à Internet.
- Conserver le port SSH `22` ouvert avant de modifier un pare-feu, pour éviter
  de perdre l'accès au serveur.
- Effectuer des sauvegardes régulières hors du VPS.
- Le site étant public, surveiller les coûts des fournisseurs de modèles et
  limiter le nombre de requêtes.
- Installer régulièrement les mises à jour de sécurité Ubuntu et Docker.

## 11. Ce qui peut encore rester à faire

Selon l'état des données actuelles :

- importer ou synchroniser la base locale vers le VPS ;
- tester une vraie question depuis l'interface ;
- mettre en place une sauvegarde automatique externe ;
- ajouter une limite de requêtes et un suivi des coûts ;
- documenter la procédure exacte du pipeline GPU vers S3 et PostgreSQL.
