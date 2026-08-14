# Comprendre le pipeline de préparation des données

Ce document explique les concepts utilisés dans `pipeline/` et montre comment
ils s'appliquent concrètement à la transformation d'une vidéo YouTube en données
exploitables par le RAG.

L'idée générale est la suivante : une vidéo et ses métadonnées sont inspectées,
une route de traitement est choisie, puis une suite de tâches produit un
transcript canonique, des intervenants, des chunks et, sur demande, des
embeddings. Les résultats peuvent ensuite être publiés dans S3 et PostgreSQL.

> **Attention au mot « contexte »** : `PipelineContext` est l'état de travail
> technique d'une vidéo pendant sa préparation. Ce n'est pas le contexte
> documentaire sélectionné par le moteur RAG pour répondre à une question.

## Vue d'ensemble

Le cycle complet des données dépasse volontairement la seule commande
`pipeline run` :

```text
YouTube
  │
  ├── vidéo + métadonnées + commentaires
  ▼
Ingestion explicite                         pipeline/ingest/
  │
  ▼
Inspection et choix de la route             PipelineContext + RoutingFacts
  │
  ▼
Plan de préparation                         planner.py
  │
  ▼
Frames / OCR / WhisperX / speakers / chunks pipeline/steps/
  │
  ├── embeddings explicites                 embeddings.create
  ▼
Publication explicite                       pipeline/publish/
  │
  ├── S3 : artefacts
  └── PostgreSQL : vidéos, transcripts, speakers, chunks et embeddings
```

Les frontières sont importantes :

- `pipeline run` inspecte et prépare une vidéo jusqu'aux chunks ;
- `embeddings.create` vectorise ensuite les chunks dans une tâche séparée ;
- l'ingestion, l'upload S3 et la synchronisation PostgreSQL sont des commandes
  explicites, en dehors du plan principal.

Cette séparation évite qu'un simple retraitement local télécharge des fichiers,
consomme l'API d'embeddings, publie des artefacts ou modifie la base de données
sans que cela ait été demandé.

## Le modèle mental en une phrase

**Un même `PipelineContext` traverse un plan ordonné de tâches ; chaque tâche
l'enrichit ou produit des artefacts, et l'état cohérent est sauvegardé dans un
manifeste après chaque transition.**

## Les concepts d'architecture

### Pipeline Pattern

Le **Pipeline Pattern** consiste à transformer une entrée par une suite d'étapes
ordonnées : la sortie d'une étape devient une entrée disponible pour les étapes
suivantes.

Ici, ce n'est pas une liste fixe de fonctions. La liste exacte dépend de la
vidéo. Par exemple, une vidéo courte avec des sous-titres incrustés passe par une
réconciliation WhisperX/OCR, alors qu'une vidéo longue évite toute l'inspection
visuelle et produit une hiérarchie de chunks.

Une étape métier ne lance jamais directement l'étape suivante. Elle produit son
résultat puis rend la main à l'exécuteur. Ce découplage permet de reprendre,
tester, remplacer ou exécuter une tâche isolément.

### Pipeline Context

`PipelineContext` est l'objet de travail unique associé à une vidéo. Il est
créé par `PipelineContext.inspect()` puis transmis à l'orchestrateur, au planner,
à l'exécuteur et à tous les handlers.

Il contient notamment :

- le chemin de la vidéo et son identifiant ;
- les caractéristiques techniques lues avec FFmpeg ;
- les métadonnées YouTube, dont la durée de référence ;
- les options effectives du lancement ;
- les faits qui déterminent la route ;
- le plan ordonné de tâches ;
- les chemins des artefacts produits par tâche ;
- l'exécution courante, ses statuts et son `run_id` ;
- l'historique des anciens plans et exécutions.

Il est **mutable par choix** : les étapes enrichissent progressivement le même
objet. Par exemple, `video.infer_type` y enregistre le type visuel et
`video.detect_subtitles` y enregistre la présence ou l'absence de sous-titres.
Cela évite de transmettre manuellement un grand nombre de paramètres entre les
étapes.

### Contrats

Un contrat donne une forme précise et validable aux échanges entre composants.
Les principaux contrats de ce pipeline sont définis dans `contracts.py` :

| Contrat | Rôle ici |
|---|---|
| `RoutingFacts` | Contient uniquement les faits persistants qui décident de la route : `video_type`, `has_subtitles` et les détails de détection. |
| `PlannedTask` | Représente une tâche choisie par le planner avec son identifiant et la raison de sa présence. |
| `TaskResult` | Décrit explicitement le résultat d'un handler : statut, raison, valeur éventuelle et artefacts. |
| `TaskExecution` | Mémorise les dates, tentatives, erreurs et empreintes d'artefacts d'une tâche. |
| `RunExecution` | Regroupe l'exécution entière sous un `run_id` et le hash exact de son plan. |

Les statuts possibles d'un `TaskResult` sont :

| Statut | Signification |
|---|---|
| `succeeded` | La tâche a produit un résultat valide. |
| `cached` | Le résultat existant est toujours valide et peut être réutilisé. |
| `skipped` | La tâche n'avait rien à faire ; ce statut n'est pas une erreur. |
| `blocked` | Une précondition ou une sortie indispensable manque ; les tâches aval sont arrêtées. |
| `failed` | Une erreur d'exécution a eu lieu ; le pipeline s'arrête. |

### Task Registry et TaskSpec

Le **Task Registry**, dans `catalog.py`, est le catalogue central des tâches
connues. Chaque `TaskSpec` associe :

- un identifiant stable, par exemple `transcript.whisper` ;
- une phase, `inspection` ou `processing` ;
- un titre lisible ;
- un handler Python ;
- une version ;
- éventuellement une postcondition vérifiable.

Le registre sait **quelle fonction** correspond à un identifiant, mais il ne
décide pas si ni quand cette tâche doit être exécutée.

La version d'une tâche participe au hash du plan. L'incrémenter permet donc
d'invalider proprement un ancien checkpoint lorsque le comportement attendu de
la tâche change.

### Planner et plan d'exécution

Le **planner**, dans `planner.py`, décide quelles tâches sont nécessaires et
dans quel ordre. Il produit une liste de `PlannedTask` ; il n'exécute rien.

Deux plans sont construits successivement :

1. `inspection_plan()` découvre les faits nécessaires au routage ;
2. `processing_plan()` prépare réellement les contenus selon ces faits.

Chaque tâche planifiée contient aussi une raison, par exemple
`reconcile_whisperx_with_ocr` ou `duration_seconds>600`. Le manifeste permet
ainsi de comprendre après coup pourquoi une tâche faisait partie du plan.

### Orchestrator

L'**orchestrateur**, dans `orchestrator.py`, coordonne le cycle de haut niveau :

1. créer ou restaurer le contexte ;
2. installer le plan d'inspection ;
3. demander son exécution ;
4. vérifier que le routage est complet ;
5. construire le plan de traitement ;
6. demander son exécution.

Il expose les comportements publics des commandes `inspect`, `plan` et `run`.
Il ne contient pas les algorithmes OCR, WhisperX ou de chunking.

### Handler

Un **handler**, dans `step_handlers.py`, est un petit adaptateur entre le monde
générique du pipeline et une fonction métier.

Sa signature conceptuelle est toujours la même :

```python
def handler(context: PipelineContext) -> TaskResult:
    ...
```

Le handler :

1. lit dans le contexte les chemins et options nécessaires ;
2. appelle la fonction métier située sous `steps/` ;
3. traduit son résultat en `TaskResult` ;
4. annonce précisément les artefacts produits.

Il ne choisit pas la tâche suivante et n'écrit pas lui-même le statut global de
l'exécution.

### Step

Une **step** est l'algorithme métier lui-même : extraction des frames, OCR,
transcription, correction, détection des speakers, chunking ou embedding.

Les steps sont rangées par domaine :

```text
pipeline/steps/
├── inspection/
├── ocr/
├── transcripts/
├── speakers/
├── chunks/
└── embeddings/
```

Cette séparation garde les algorithmes utilisables et testables sans leur faire
connaître toute la mécanique d'orchestration.

### Executor

L'**exécuteur**, dans `executor.py`, fait avancer le plan tâche par tâche. Pour
chaque identifiant, il récupère la `TaskSpec` dans le registre, vérifie si un
checkpoint peut être repris, appelle sinon le handler, valide son `TaskResult`
et sauvegarde le manifeste.

Il est aussi responsable des règles d'arrêt :

- `blocked` marque les tâches aval comme `skipped` puis bloque le run ;
- `failed` marque le run en échec et propage l'erreur ;
- `succeeded`, `cached` et `skipped` permettent de continuer.

### Artefact

Un **artefact** est un fichier ou dossier produit par une tâche : frames, JSON
OCR, transcript, fichier de speakers, chunks ou embeddings.

Le contexte conserve les chemins par identifiant de tâche dans
`artifacts.by_task`. L'exécuteur peut ainsi vérifier qu'un résultat déclaré
existe encore et n'a pas changé avant de le reprendre depuis le cache.

### Manifest et checkpointing

Le **manifest**, `metadata/video_manifest.json`, est la représentation persistée
du contexte. C'est la source de vérité de l'orchestration ; il ne faut pas le
modifier manuellement.

Un **checkpoint** est une sauvegarde de ce manifeste à un moment cohérent. Il
est écrit atomiquement :

- après la construction d'un plan ;
- avant chaque handler, avec la tâche en `running` ;
- après chaque handler, avec son statut et ses artefacts ;
- immédiatement lors d'un blocage ou d'une exception ;
- à la fin, avec le run en `completed`.

Le manifeste contient notamment :

```text
schema_version
video
routing_facts
route
options
artifacts.by_task
plan.hash
plan.tasks
execution.run_id
execution.tasks
execution_history
```

Le **hash du plan** tient compte des tâches, de leurs raisons, versions et
handlers, des options pertinentes, des faits de routage du traitement et de
l'empreinte du fichier vidéo. `force` n'en fait pas partie, car c'est une
instruction ponctuelle et non une propriété des données attendues.

Une tâche réussie n'est reprise que si sa postcondition est encore vraie ou si
ses artefacts existent toujours avec une empreinte compatible. Dès qu'une tâche
doit être rejouée, les tâches situées après elle sont également rejouées pour
éviter de mélanger des sorties nouvelles et anciennes.

## Comment les données sont préparées ici

### 1. Acquisition des entrées

L'ingestion crée ou met à jour, pour chaque vidéo, les sources suivantes :

```text
downloads/youtube/init/VIDEO_ID/
├── VIDEO_ID.mp4
└── metadata/
    ├── youtube_video_metadata.json
    └── youtube_comments.json
```

Ces éléments ont des rôles distincts :

| Source | Utilisation |
|---|---|
| Fichier vidéo | Image et audio ; FFmpeg lit aussi codecs, résolution, FPS, rotation et présence d'une piste audio. |
| `youtube_video_metadata.json` | Titre et durée de référence utilisée pour choisir la stratégie courte ou longue. |
| `youtube_comments.json` | Commentaires et réponses, conservés pour la publication SQL ; ils ne pilotent pas le plan de préparation audiovisuelle. |

La durée utilisée pour le routage vient des métadonnées YouTube. La vidéo est
considérée longue si sa durée est **strictement supérieure à 600 secondes** ;
une durée exactement égale à 600 secondes reste courte.

### 2. Création ou restauration du contexte

`PipelineContext.inspect()` :

1. résout le fichier vidéo ;
2. recharge le manifeste existant s'il existe ;
3. charge les métadonnées YouTube ;
4. sonde le fichier avec FFmpeg ;
5. restaure les faits de routage, artefacts, options et exécutions précédentes ;
6. impose `long_video` lorsque la durée dépasse 600 secondes.

À ce stade, le contexte contient les sources et l'historique connus, mais les
faits visuels peuvent encore manquer.

### 3. Inspection et routage

Pour une vidéo courte, le plan d'inspection est :

```text
frames.extract
frames.classify
video.detect_interview
video.infer_type
ocr.extract_raw
ocr.extract_boxes
video.detect_subtitles
```

Cette phase répond principalement à deux questions :

- quel est le type visuel de la vidéo : `interview`, `motion_design` ou
  `video_recording` ?
- contient-elle des sous-titres incrustés exploitables ?

Pour une vidéo longue, la durée suffit à imposer `long_video`. Le plan conserve
seulement `video.infer_type` et évite l'extraction des frames, la classification,
l'OCR et la détection des sous-titres.

Le résultat persistant est un `RoutingFacts`. La route dérivée prend la forme :

```text
<chunk_strategy>.<transcript_strategy>.<visual_strategy>
```

Exemples :

```text
short.whisper.interview
short.whisper.motion_design
long.whisper.long_video
```

La stratégie canonique de transcript est toujours `whisper`. La présence de
sous-titres ne remplace donc pas WhisperX par l'OCR : elle active une référence
OCR destinée à corriger WhisperX.

### 4. Traitement d'une vidéo courte sans sous-titres incrustés

Le plan est conceptuellement :

```text
ocr.build_processed
ocr.filter_overlays
transcript.whisper
transcript.correct_whisper
speakers.propose
speakers.validate
transcript.enrich
transcript.create_plain
chunks.create
```

L'OCR visuel nettoyé peut aider à corriger le transcript WhisperX selon le mode
de correction configuré, mais aucun transcript OCR de sous-titres n'est créé.
Le transcript enrichi reçoit ensuite les speakers validés et les intercalaires
visuels pertinents.

### 5. Traitement d'une vidéo courte avec sous-titres incrustés

Le plan ajoute une branche OCR de correction :

```text
ocr.build_processed
ocr.filter_overlays
transcript.whisper
transcript.extract_ocr
transcript.normalize_brand
transcript.create_plain_ocr
transcript.reconcile_ocr
speakers.propose
speakers.validate
transcript.enrich
transcript.create_plain
chunks.create
```

Trois données doivent être distinguées :

| Donnée | Rôle |
|---|---|
| `transcript_1_brut.txt` | Sortie brute de WhisperX, conservée comme trace. |
| `transcripts_ocr/plain_transcript.txt` | Référence OCR interne, utilisée seulement pour corriger WhisperX. |
| `transcript_2_corrected.txt` | Transcript canonique corrigé par rapprochement avec l'OCR. |

La réconciliation corrige le texte de chaque segment, tandis que le code garde
la maîtrise des timecodes, de l'ordre, du nombre de segments et des identifiants
de speakers.

Le transcript OCR n'est consommé ni par le RAG ni par la publication SQL. Tous
les consommateurs finaux utilisent la branche canonique
`outputs/transcripts_whisper/`.

### 6. Traitement d'une vidéo longue

Le plan est volontairement plus léger :

```text
transcript.whisper
transcript.create_plain
speakers.propose
speakers.validate
chunks.create
chunks.summarize_sections
chunks.summarize_video
```

Il n'y a ni inspection visuelle complète, ni OCR, ni correction visuelle, ni
transcript enrichi. Le transcript plain est créé directement à partir du
WhisperX brut. Les speakers validés restent un artefact publiable séparé.

### 7. Construction du transcript canonique

Pour une vidéo courte, les artefacts canoniques sont rangés sous
`outputs/transcripts_whisper/` :

| Artefact | Contenu |
|---|---|
| `transcript_1_brut.txt` | Segments WhisperX bruts. |
| `transcript_2_corrected.txt` | Texte corrigé tout en conservant la structure temporelle. |
| `transcript_3_enriched.txt` | Transcript corrigé avec speakers validés et intercalaires pertinents. |
| `transcript_plain.txt` | Texte final sans timecodes, utilisé pour créer les chunks. |

Pour une vidéo longue, seuls `transcript_1_brut.txt` et
`transcript_plain.txt` sont produits.

Les speakers validés sont stockés séparément dans
`outputs/speakers/speakers_validated.json`. Ils ne sont pas dupliqués dans les
chunks.

### 8. Création des chunks

`chunks.create` lit le `transcript_plain.txt` canonique, normalise les lignes et
découpe le texte aux frontières de phrases. La taille cible actuelle est de
1 000 caractères ; la coupure se fait à la fin de la phrase qui atteint ou
dépasse ce seuil.

Le résultat principal est :

```text
outputs/chunks/transcript_chunks.json
```

Pour une vidéo courte, il contient uniquement des chunks `detail`.

Pour une vidéo longue, le traitement ajoute deux niveaux de synthèse :

```text
global
└── section
    └── detail
```

- `detail` contient les passages du transcript ;
- `section` résume un groupe de chunks détail, six par défaut ;
- `global` résume les sections de la vidéo.

Les relations logiques sont conservées dans le JSON puis converties en
`chunk_parent_id` pendant la synchronisation PostgreSQL.

### 9. Création explicite des embeddings

Les embeddings ne font pas partie de `processing_plan()`. Ils sont créés après
les chunks avec :

```powershell
.\.venv\Scripts\python.exe -m pipeline task embeddings.create VIDEO_ID
```

Chaque chunk non vide est vectorisé. La configuration actuelle utilise
`text-embedding-3-large` avec 2 000 dimensions. Les fichiers produits sont
rangés avec les chunks, par exemple :

```text
outputs/chunks/chunk_01_embedding.json
outputs/chunks/chunk_section_01_embedding.json
outputs/chunks/chunk_global_01_embedding.json
```

Le contenu, le modèle et le nombre de dimensions sont enregistrés avec le
vecteur. Une relance peut ainsi détecter un embedding obsolète et le régénérer.

### 10. Publication

La publication est également explicite :

```powershell
.\.venv\Scripts\python.exe -m pipeline.publish.upload_outputs_to_s3
.\.venv\Scripts\python.exe -m pipeline.publish.sync_database
```

L'upload S3 publie les artefacts. La synchronisation PostgreSQL importe les
métadonnées, commentaires, transcripts, speakers, chunks et embeddings dans le
schéma relationnel utilisé par l'interface RAG.

## Arborescence finale typique

Selon la route, une vidéo préparée ressemble à ceci :

```text
VIDEO_ID/
├── VIDEO_ID.mp4
├── metadata/
│   ├── youtube_video_metadata.json
│   ├── youtube_comments.json
│   └── video_manifest.json
└── outputs/
    ├── images/                    # inspection des vidéos courtes
    ├── interview/                 # indices de détection d'interview
    ├── ocr/                       # OCR brut, positions et données nettoyées
    ├── transcripts_ocr/           # référence OCR de correction, si applicable
    ├── transcripts_whisper/       # branche canonique
    ├── speakers/
    │   └── speakers_validated.json
    └── chunks/
        ├── transcript_chunks.json
        └── chunk_*_embedding.json # après embeddings.create
```

Tous ces dossiers ne sont pas attendus sur toutes les routes. Une vidéo longue,
par exemple, ne produit normalement ni `images/`, ni OCR, ni transcript OCR.

## Qui décide quoi ?

| Composant | Question à laquelle il répond |
|---|---|
| `PipelineContext` | Qu'est-ce que l'on sait actuellement de cette vidéo et de son exécution ? |
| `RoutingFacts` | Quels faits persistants permettent de choisir la route ? |
| `planner.py` | Quelles tâches faut-il exécuter, et dans quel ordre ? |
| `catalog.py` | Quel handler correspond à cet identifiant de tâche ? |
| `step_handlers.py` | Comment adapter le contexte à l'algorithme métier et décrire son résultat ? |
| `steps/` | Comment effectuer concrètement le traitement ? |
| `executor.py` | Faut-il reprendre le cache ou appeler le handler, et peut-on continuer ? |
| `orchestrator.py` | Comment coordonner inspection, planification et traitement ? |
| `manifest.py` | Comment valider et persister l'état cohérent du pipeline ? |

Trois règles aident à retenir la séparation :

1. le registre sait **quoi appeler**, pas **quand** ;
2. le planner décide **quoi ordonner**, pas **comment le calculer** ;
3. une step produit sa sortie, mais ne déclenche jamais directement la suivante.

## Commandes utiles pour comprendre un cas réel

Afficher le plan sans exécuter les traitements :

```powershell
.\.venv\Scripts\python.exe -m pipeline plan VIDEO_ID
```

Inspecter la vidéo et calculer ses faits de routage :

```powershell
.\.venv\Scripts\python.exe -m pipeline inspect VIDEO_ID
```

Afficher les handlers qui seraient appelés :

```powershell
.\.venv\Scripts\python.exe -m pipeline run VIDEO_ID --skip-inspection --dry-run
```

Préparer effectivement la vidéo jusqu'aux chunks :

```powershell
.\.venv\Scripts\python.exe -m pipeline run VIDEO_ID
```

Lister le registre ou relancer une seule tâche :

```powershell
.\.venv\Scripts\python.exe -m pipeline task --list
.\.venv\Scripts\python.exe -m pipeline task chunks.create VIDEO_ID
```

Forcer un run complet supprime d'abord le dossier `outputs/`, puis reconstruit
la chaîne principale jusqu'aux chunks :

```powershell
.\.venv\Scripts\python.exe -m pipeline run VIDEO_ID --force
```

Les embeddings ayant été supprimés avec `outputs/`, ils doivent ensuite être
recréés explicitement.

## Comment lire un problème de pipeline

En cas de résultat inattendu, l'ordre de diagnostic le plus utile est :

1. vérifier la vidéo et `metadata/youtube_video_metadata.json` ;
2. lire `routing_facts` et `route` dans `video_manifest.json` ;
3. regarder `plan.tasks` pour connaître la route réellement planifiée ;
4. inspecter `execution.tasks.<task_id>` pour trouver le premier statut non
   réussi et son message ;
5. vérifier dans `artifacts.by_task` les fichiers annoncés par cette tâche ;
6. remonter ensuite à son handler dans `step_handlers.py`, puis à son algorithme
   sous `steps/`.

Ce chemin suit exactement les frontières de l'architecture et évite de chercher
un problème de routage dans un algorithme métier, ou un problème d'artefact dans
le planner.

## Résumé à retenir

- **Pipeline Context** : l'état vivant d'une vidéo pendant sa préparation.
- **RoutingFacts** : les faits persistants qui déterminent la route.
- **Planner** : la construction de la liste ordonnée de tâches.
- **Task Registry** : la correspondance entre identifiants et handlers.
- **Handler** : l'adaptateur entre le contexte et l'algorithme métier.
- **Step** : le traitement métier réel.
- **Executor** : l'exécution, la reprise et les règles d'arrêt.
- **Manifest** : la version persistée et auditable du contexte.
- **Checkpoint** : une sauvegarde cohérente entre deux transitions.
- **Artefact** : une sortie concrète produite par une tâche.

Le point essentiel est que la préparation n'est pas une succession opaque de
scripts. C'est un plan explicite, calculé à partir de faits vérifiables, exécuté
avec des contrats typés et sauvegardé après chaque étape.
