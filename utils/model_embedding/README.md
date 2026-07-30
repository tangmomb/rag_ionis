# Benchmark des modèles d'embedding

Ce dossier compare plusieurs modèles et dimensions sur les chunks réellement produits par le pipeline. Il ne modifie ni PostgreSQL, ni les fichiers `chunk_XX_embedding.json` de production.

## Contenu

- `configs.json` : modèles, dimensions et valeurs de `top_k` à comparer ;
- `cases.example.jsonl` : exemples de questions annotées ;
- `cases.jsonl` : jeu d'évaluation local à créer ;
- `benchmark.py` : catalogue, appels OpenAI, cache, classement cosinus et rapports ;
- `generate_cases.py` : sélection équilibrée de chunks et génération d'un brouillon de questions ;
- `test_benchmark.py` : tests unitaires sans appel réseau ;
- `cache/` : embeddings calculés, ignorés par Git ;
- `reports/` : résultats générés, ignorés par Git.

## 1. Générer un brouillon de questions

Commencer par exporter le catalogue des chunks, sans appel OpenAI :

```powershell
python utils/model_embedding/benchmark.py `
  --catalog-only
```

Le fichier `reports/chunk_catalog.csv` contient les clés stables au format `video_key:chunk_index` et le texte de chaque chunk.

Vérifier d'abord la sélection automatique sans appel OpenAI :

```powershell
python utils/model_embedding/generate_cases.py `
  --video-dir downloads/youtube `
  --selection-only
```

Le fichier `reports/generation_selection.csv` répartit les chunks en round-robin entre les vidéos. Générer ensuite les questions :

```powershell
python utils/model_embedding/generate_cases.py `
  --video-dir downloads/youtube
```

Le résultat est écrit dans `cases.generated.jsonl`. Chaque question est rattachée automatiquement au chunk utilisé pour la produire :

```json
{"id":"q001","question":"Quel poste occupe Alice ?","category":"prenom_court","relevant":[{"video_key":"abc123","chunk_index":7}]}
```

Le générateur crée deux groupes dans cet ordre :

- `q001` à `q025`, catégorie `prenom_court` : question de 16 mots maximum utilisant uniquement le prénom du speaker ;
- `q026` à `q050`, catégorie `contenu_courant` : question simple sur le contenu, sans mentionner le speaker.

Relire chaque ligne et vérifier que :

- la réponse est explicitement présente dans le chunk indiqué ;
- la question est naturelle et compréhensible seule ;
- elle ne recopie pas simplement une phrase du chunk ;
- elle n'est ni ambiguë, ni trop facile, ni fondée sur une information extérieure.

Après validation, copier le brouillon puis supprimer ou corriger les mauvaises lignes :

```powershell
Copy-Item utils/model_embedding/cases.generated.jsonl utils/model_embedding/cases.jsonl
```

Le fichier `cases.generated.manifest.json` conserve le modèle utilisé, la graine, le nombre de vidéos, le hash des sources et les tokens d'entrée. Les fichiers générés sont ignorés par Git ; le fichier final `cases.jsonl` peut être versionné.

Options du générateur :

```text
--model          modèle de génération, gpt-5.6-luna par défaut
--person-count   questions courtes au prénom, 25 par défaut
--content-count  questions courantes sans speaker, 25 par défaut
--seed           graine de sélection, 42 par défaut
--batch-size     chunks par appel, 8 par défaut
--min-chars      longueur minimale d'un chunk, 120 par défaut
--max-per-video  limite par vidéo, 0 signifie sans limite
--force          remplace un brouillon existant
```

## 2. Cas particuliers

Une question sans réponse dans le corpus s'écrit ainsi :

```json
{"id":"q099","question":"Question absente du corpus","relevant":[],"answerable":false}
```

Les questions non répondables sont conservées dans le rapport détaillé, mais exclues de Recall@K et MRR : une recherche vectorielle seule renvoie toujours des voisins et ne suffit pas à mesurer le refus de répondre.

## 3. Lancer la comparaison

La variable `OPENAI_API_KEY` doit être présente dans l'environnement ou dans `.env` à la racine du projet.

```powershell
python utils/model_embedding/benchmark.py
```

Options utiles :

```text
--cases       chemin du JSONL annoté
--configs     configurations à comparer
--batch-size  nombre de textes envoyés par appel, 64 par défaut
--cache-dir   dossier du cache
--reports-dir dossier des rapports
```

Le premier lancement calcule les embeddings. Les lancements suivants réutilisent le cache tant que le modèle, la dimension et les textes n'ont pas changé.

## 4. Lire les résultats

- `reports/report.md` : rapport Markdown unique organisé en chapitres avec résumé global, catégories, définitions, échecs et détail par question.

La taille `bytes_per_vector` correspond au stockage brut pgvector `vector`, soit `4 × dimensions + 8`. Elle n'inclut pas les lignes PostgreSQL ni un éventuel index HNSW.

Une règle de choix raisonnable est de retenir la configuration la moins chère dont le Recall@10 reste à moins de deux points du meilleur résultat, puis de comparer les deux finalistes dans le pipeline BM25 + vectoriel + RRF.

## 5. Exécuter les tests unitaires

```powershell
python -m unittest utils.model_embedding.test_benchmark -v
```

Ces tests n'utilisent ni `OPENAI_API_KEY`, ni PostgreSQL.

## 6. Benchmark RAG hybride

Après le benchmark vectoriel, comparer les finalistes avec la recherche réelle :

```powershell
python utils/model_embedding/benchmark_rag.py
```

La configuration est définie dans `rag_configs.json`. Le test compare actuellement `small-1024`, `large-1024`, `large-2000` et `large-3072` avec :

```text
recherche lexicale PostgreSQL (`ts_rank_cd`) top 40
recherche vectorielle top 40
fusion RRF top 30
reranker Cohere top 5
```

`rerank_min_interval_seconds` vaut `6.2` par défaut dans ce projet afin de respecter la limite de 10 appels par minute d'une clé Cohere d'essai. Une clé de production peut utiliser `0`.

Le test lit `data.chunks` et `data.videos` sans les modifier. Les embeddings viennent du cache créé par `benchmark.py`. Les résultats Cohere sont mis en cache dans `cache/rag_rerank/`.

La recherche lexicale reproduit le `ts_rank_cd` PostgreSQL actuellement utilisé par l'application ; ce n'est pas un BM25 strict. Le classement vectoriel est exact et effectué en mémoire pour comparer la pertinence des dimensions sans être bloqué par la limite HNSW de `vector(3072)`. Sa latence ne mesure donc pas pgvector ni HNSW.

Le rapport unique est écrit dans `reports/rag_report.md`. Il compare Recall@K, MRR et latence après chaque étape, globalement et par catégorie.

Le planner, les préfiltres métier et le LLM de réponse sont volontairement exclus afin d'isoler l'effet du modèle d'embedding sur le retrieval hybride.
