# Phoenix et OpenTelemetry dans RAG IONIS

## Leur rôle en une phrase

**OpenTelemetry produit et transporte les traces techniques du RAG ; Phoenix les stocke et fournit l’interface graphique pour les explorer.**

Dans ce projet, ils servent à comprendre ce qui se passe pendant un appel à `POST /api/rag` : plan choisi, recherches exécutées, sources récupérées, génération finale, durées et erreurs.

Ils n’exécutent pas le RAG et ne remplacent pas PostgreSQL.

Pour le pipeline batch de préparation des vidéos, voir [PREFECT.md](PREFECT.md).

## Répartition des responsabilités

| Composant | Responsabilité dans le projet |
|---|---|
| OpenTelemetry | Créer une trace et des spans autour des étapes Python |
| OpenInference | Donner un vocabulaire adapté aux applications IA : `CHAIN`, `AGENT`, `RETRIEVER`, `RERANKER`, etc. |
| Instrumentation OpenAI | Tracer automatiquement les appels effectués avec le client OpenAI |
| Phoenix | Recevoir, conserver et afficher les traces dans une interface graphique |
| PostgreSQL `chat.messages` | Conserver durablement la conversation, la réponse et les données métier choisies par l’application |

Une **trace** correspond normalement à une requête RAG complète. Un **span** correspond à une opération à l’intérieur de cette requête.

## Flux dans le projet

```mermaid
flowchart LR
    U["Utilisateur"] --> API["POST /api/rag"]
    API --> S["Spans OpenTelemetry"]
    S --> O["Reformulation, planner, retrieval, génération"]
    O --> SQL["Écriture dans chat.messages"]
    S -->|"HTTP/protobuf"| P["Phoenix : localhost:6006"]
    P --> UI["Chronologie et détails graphiques"]
    SQL --> T["Message, réponse, traces JSONB et trace_id"]
```

Au démarrage de FastAPI, `configure_telemetry()` :

1. lit les variables `PHOENIX_*` ;
2. enregistre un fournisseur de traces OpenTelemetry ;
3. configure l’envoi vers Phoenix en HTTP/protobuf ;
4. active l’instrumentation automatique du client OpenAI ;
5. laisse l’API fonctionner même si Phoenix est indisponible.

Le code principal se trouve dans [`interface/backend/telemetry.py`](../interface/backend/telemetry.py).

## Ce qui est visible dans Phoenix

Les spans manuels actuels sont notamment :

| Span | Ce qu’il permet d’inspecter |
|---|---|
| `rag.request` | Requête complète, conversation, modèles demandés et réponse finale |
| `rag.orchestration` | Résultat global du routage et de la récupération |
| `rag.reformulation` | Question initiale, contexte conversationnel et question reformulée |
| `rag.planner` | Prompt du planner, sortie brute, validation Pydantic et plan retenu |
| `rag.speaker_resolution` | Noms demandés, correspondances et ambiguïtés |
| `rag.memory` | Lecture de l’historique de conversation |
| `rag.structured_sql` | Recherche structurée d’une vidéo ou de ses métadonnées |
| `rag.retrieval.prefilter` | Filtres SQL et identifiants candidats |
| `rag.retrieval.embedding` | Modèle d’embedding et nombre de dimensions |
| `rag.retrieval.bm25` | Requête lexicale, scores et chunks obtenus |
| `rag.retrieval.vector` | Recherche vectorielle et résultats |
| `rag.retrieval.rrf` | Fusion des classements BM25 et vectoriel |
| `rag.retrieval.rerank` | Entrées et résultats du reranking Cohere |
| `rag.source_evaluation` | Évaluation de la suffisance des sources |
| `rag.generation` | Sources envoyées au modèle et réponse générée |
| `rag.store_message` | Identifiants de conversation et de message écrits en SQL |

Pour chaque span, Phoenix peut afficher selon l’étape :

- les entrées et sorties JSON ;
- les attributs techniques ;
- la durée et l’ordre d’exécution ;
- les sous-spans OpenAI instrumentés automatiquement ;
- les erreurs levées pendant l’opération ;
- l’identifiant de session, qui correspond à `conversation_id` quand il est disponible.

## Relation avec la table `chat.messages`

Phoenix et `chat.messages` contiennent des informations qui se recouvrent, mais ils n’ont pas le même objectif.

| Besoin | Phoenix | `chat.messages` |
|---|---:|---:|
| Voir la chronologie imbriquée des étapes | Oui | Non, sauf reconstruction manuelle |
| Comparer les durées | Oui | Très limité |
| Inspecter rapidement prompts, résultats et erreurs | Oui | Oui pour les champs explicitement enregistrés |
| Relire la conversation dans l’application | Non | Oui |
| Garantir le stockage métier selon le schéma de l’application | Non | Oui |
| Faire des requêtes SQL ou des exports métier | Non | Oui |

Le champ `chat.messages.trace_id` contient l’identifiant hexadécimal de la trace Phoenix. Il permet de relier un message SQL à son exécution technique.

Pour l’instant, les colonnes détaillées comme `planner_prompt`, `bm25_trace`, `vector_trace` et `answer_response_raw` sont toujours conservées. Il est préférable de vérifier la parité sur des requêtes réelles avant de retirer certaines de ces colonnes. Même après cette vérification, les champs métier — question, réponse, sources citées, conversation et `trace_id` — doivent rester en SQL si l’application en dépend.

## Démarrage et accès

Le moyen habituel de tout démarrer est :

```powershell
.\start_app.bat
```

Pour démarrer seulement PostgreSQL et Phoenix :

```powershell
docker compose up -d postgres phoenix
```

Interfaces et contrôles :

- Phoenix : <http://127.0.0.1:6006/>
- état de la télémétrie : <http://127.0.0.1:8006/version>
- état de l’API : <http://127.0.0.1:8006/health>

Dans Phoenix, sélectionner le projet `rag-ionis`, puis ouvrir une trace `rag.request` pour développer ses sous-spans.

## Configuration

```dotenv
PHOENIX_ENABLED=true
PHOENIX_PORT=6006
PHOENIX_GRPC_PORT=4317
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_PROJECT_NAME=rag-ionis
```

Le projet utilise actuellement l’endpoint HTTP `6006/v1/traces`. Le port gRPC `4317` est exposé par Docker mais n’est pas utilisé par `interface/backend/telemetry.py`.

Les données Phoenix persistent dans le volume Docker `rag_ionis_phoenix_data`.

Le backend utilise de préférence `.venv-interface`, car les dépendances OpenTelemetry/Phoenix sont isolées de l’environnement GPU du pipeline.

## Lire une trace pour diagnostiquer une mauvaise réponse

Ordre conseillé :

1. ouvrir `rag.request` et vérifier la question et le modèle ;
2. lire `rag.reformulation` pour détecter une mauvaise reprise de contexte ;
3. lire `rag.planner` et son `execution_plan` ;
4. contrôler le préfiltre et les résultats BM25/vectoriels ;
5. vérifier l’ordre après RRF et Cohere ;
6. ouvrir `rag.source_evaluation` ;
7. comparer les chunks disponibles avec ceux réellement cités dans `rag.generation` ;
8. utiliser le `trace_id` pour retrouver le message correspondant en SQL.

Cette lecture permet de distinguer une erreur de planification, de retrieval, de ranking ou de génération.

## Ajouter une nouvelle opération tracée

```python
from interface.backend.telemetry import trace_operation

with trace_operation(
    "rag.ma_nouvelle_etape",
    kind="CHAIN",
    input_value={"question": question},
) as span:
    result = ma_nouvelle_etape(question)
    span.set_attribute("rag.result_count", len(result))
    span.set_output(result)
```

Le contexte OpenTelemetry rattache automatiquement ce span au span parent actif.

## Limites et précautions

- Phoenix n’est pas la base métier de l’application.
- Les traces peuvent contenir questions, prompts, chunks et réponses : l’interface est donc liée uniquement à `127.0.0.1`.
- Une sortie JSON très volumineuse rend l’interface plus lente ; mieux vaut tracer des aperçus ou des compteurs quand le volume augmente.
- Si Phoenix tombe, le RAG continue normalement : l’observabilité a été conçue comme non bloquante.
- Désactiver l’envoi avec `PHOENIX_ENABLED=false` si aucune trace ne doit être produite.

## Dépannage rapide

```powershell
docker compose ps phoenix
docker compose logs --tail 100 phoenix
Invoke-RestMethod http://127.0.0.1:8006/version
```

Dans `/version` :

- `configured: true` signifie que l’initialisation a été tentée ;
- `enabled: true` signifie que le tracer est actif ;
- `initialization_error` explique pourquoi l’initialisation a échoué, le cas échéant.
