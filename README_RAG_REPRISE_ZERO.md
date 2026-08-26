# Reprise à zéro — fiabilisation du RAG IONIS

Ce document est le cahier des charges pour une future reprise du RAG. Il part
volontairement du principe qu'aucune des améliorations décrites ci-dessous n'a
encore été développée.

La trace Phoenix de référence à reproduire avant toute modification est :

```text
39778ed8c7d2a20c6be785f59655e996
```

Le défaut principal à corriger est la dégradation des réponses lorsque la
conversation s'allonge, en particulier pendant une recherche successive de
plusieurs personnes ou vidéos suivie d'une question de comparaison.

## Objectifs non négociables

- Une relance doit conserver uniquement le contexte utile du sujet actif.
- Un nom explicitement écrit par l'utilisateur ne doit jamais hériter du
  patronyme d'une personne précédemment mentionnée.
- Une comparaison de plusieurs personnes doit utiliser l'union de leurs vidéos,
  jamais une intersection implicite.
- Chaque affirmation factuelle doit être rattachée à une preuve identifiable.
- L'absence de preuve doit conduire à une abstention claire, pas à une
  généralisation plausible.
- Une erreur fournisseur, notamment HTTP 429, ne doit pas produire une réponse
  non vérifiée ni une erreur HTTP 500 évitable.
- La qualité doit être mesurée sur de vraies conversations multi-tours, pas
  seulement sur des fonctions isolées.

## 1. Établir la référence avant de modifier le code

1. Rejouer la trace Phoenix de référence avec
   `utils/replay_phoenix_conversation.py`.
2. Exporter pour chaque tour : question originale, reformulation, plan,
   personnes résolues, identifiants vidéo, SQL, chunks retenus, réponse,
   latence et erreurs fournisseur.
3. Transformer ce scénario en cas golden multi-tours reproductible.
4. Ajouter des cas voisins : changement de sujet, correction d'un nom,
   ambiguïté réelle, oubli explicite d'une ancienne vidéo et conversation de
   plus de dix tours.

La première passe est uniquement diagnostique. Elle ne doit modifier ni les
données ni les prompts.

## 2. Borner et structurer la mémoire conversationnelle

La reformulation ne doit pas recevoir toute la conversation brute.

À mettre en place :

- une fenêtre courte des derniers échanges du sujet actif ;
- une mémoire structurée séparant `persons`, `companies`, `videos`, dernier
  objectif autonome, corrections et ordre de récence ;
- un identifiant de sujet pour isoler un nouveau sujet de l'ancien contexte ;
- des événements explicites `added`, `removed`, `corrected` et `forgotten` ;
- une limite en nombre de tours, caractères et entités ;
- une politique déterministe pour les relances elliptiques comme « celle
  d'Oussama », « ajoute Déborah » ou « ont-elles des points communs ? ».

Le texte explicitement saisi au tour courant est prioritaire sur la mémoire et
sur le LLM. Un prénom isolé doit être envoyé au résolveur SQL sans ajout inventé.
Le développement d'un prénom vers un nom complet appartient au résolveur de la
base, pas au modèle de reformulation.

Tests obligatoires :

- Fadila puis « Et celle d'Oussama » résout Oussama Bouziza, jamais Oussama
  Ouro Sama ;
- une correction retire l'ancienne personne de la mémoire active ;
- un nouveau sujet n'importe aucune ancienne vidéo ;
- « les trois vidéos » retrouve exactement les trois identifiants canoniques ;
- le volume du prompt reste borné quand la conversation s'allonge.

## 3. Résoudre les entités avant la recherche

Les titres et noms libres ne doivent pas servir de clés durables.

Le flux cible est :

```text
question → mentions explicites → résolution SQL → video_ids canoniques → recherche
```

À mettre en place :

- résolution exacte normalisée, puis résolution unique par prénom ;
- fuzzy matching uniquement avec seuil, marge entre candidats et trace du
  score ;
- clarification si plusieurs personnes restent plausibles ;
- résolution des entreprises séparée de celle des personnes ;
- stockage des `video_id`, titres et URL canoniques dans la mémoire ;
- déduplication par identifiant vidéo, puis par identifiant YouTube normalisé.

Une requête contenant plusieurs personnes doit produire l'union des vidéos
associées. Une recherche d'intersection n'est autorisée que si l'utilisateur
demande explicitement leur présence commune dans une même vidéo.

## 4. Séparer clairement SQL, RAG et comparaison multi-source

Le planner doit produire un contrat typé et validé, sans répondre à la question.

Routes attendues :

- `direct` pour les messages sociaux sans recherche ;
- `rag` pour une question sur le contenu d'une ou plusieurs vidéos ;
- `multi_source` lorsqu'une comparaison ou une synthèse croisée est requise.

Sous-intentions SQL attendues :

- recherche de personnes ou vidéos précises ;
- statistiques et métadonnées structurées ;
- description ;
- transcript intégral explicitement demandé ;
- aucune sous-intention SQL lorsque la réponse dépend seulement du contenu des
  transcripts.

Les règles déterministes doivent corriger les erreurs fréquentes du planner :
route sociale trop large, faux titre extrait d'une formulation générique,
intersection de personnes et SQL demandé pour un fait purement sémantique.

## 5. Construire une recherche hybride et hiérarchique

La recherche cible combine :

1. préfiltrage SQL par identifiants vidéo canoniques ;
2. recherche lexicale BM25 ;
3. recherche vectorielle ;
4. fusion RRF ;
5. reranking ;
6. diversification finale par vidéo.

Pour les vidéos longues, conserver trois niveaux de chunks :

- `detail` pour citer une affirmation précise ;
- `section` pour retrouver un thème local ;
- `global` pour orienter la recherche dans la vidéo.

Les résultats `section` et `global` servent de contexte de navigation. Une
affirmation finale doit, autant que possible, être appuyée par un chunk `detail`.

## 6. Traiter les comparaisons avec une matrice vidéo × axe

Une comparaison ne doit pas rechercher une seule phrase supposée commune aux
documents. Elle doit décomposer la question en axes stables, par exemple :

- parcours et formation ;
- missions et responsabilités ;
- compétences, motivations ou conseils.

Pour chaque couple `(video_id, axe)`, exécuter une recherche bornée et produire
un registre déterministe contenant : identifiant vidéo, axe, chunks candidats,
score et état de couverture.

La présence d'un chunk candidat signifie uniquement qu'une preuve potentielle a
été trouvée. Elle ne prouve pas que les vidéos partagent le même fait.

Après génération et vérification, produire un second registre des preuves
validées. Chaque fait commun doit indiquer s'il couvre toutes les vidéos ou
seulement un sous-ensemble. S'il n'existe aucun point commun solide à toutes les
vidéos, la réponse doit le dire puis présenter les similitudes partielles sans
les généraliser.

## 7. Vérifier les affirmations et échouer de façon fermée

Ajouter deux contrôles distincts :

1. un vérificateur de claims qui décompose la réponse en affirmations et associe
   chacune à ses sources ;
2. un juge sémantique qui contrôle l'intention, la couverture, le SQL et la
   cohérence globale de la réponse.

Un verdict est positif uniquement si les deux contrôles sont positifs. En cas
d'échec :

- corriger une seule fois l'étape réellement fautive : SQL, retrieval ou
  génération ;
- vérifier à nouveau la réponse corrigée ;
- si des claims restent non étayés, construire une réponse uniquement à partir
  des claims validés ;
- si rien de fiable ne reste, retourner `abstain`.

Une exception du juge ou du vérificateur doit être considérée comme un contrôle
échoué. Elle ne doit jamais rendre implicitement la réponse valide.

## 8. Rendre les appels fournisseurs résistants aux erreurs

Centraliser la politique des appels LLM, embeddings et reranking :

- timeout explicite ;
- détection normalisée des statuts 408, 409, 425, 429 et 5xx temporaires ;
- backoff exponentiel avec jitter ;
- nombre maximal de tentatives par appel ;
- budget global de retries par requête ;
- circuit breaker court par fournisseur après épuisement sur 429 ;
- réponse déterministe ou abstention quand la génération n'est plus disponible.

Les SDK ne doivent pas appliquer des retries cachés en plus de cette politique.
Chaque tentative, délai, 429 et ouverture de circuit doit apparaître dans la
trace.

## 9. Réduire la latence et le coût

Pour une matrice de trois vidéos par trois axes, ne pas effectuer neuf appels
d'embedding identiques.

À mettre en place :

- déduplication et batch des requêtes d'embedding ;
- cache borné et expirant par modèle, dimensions et texte normalisé ;
- réutilisation d'un embedding dans toutes les cellules concernées ;
- parallélisme borné des recherches indépendantes ;
- limitation et diversification du pool envoyé au reranker ;
- arrêt anticipé lorsqu'une étape déterministe suffit.

Mesurer par requête : latence totale, latence par opération, nombre d'appels,
retries, 429, tokens d'entrée et de sortie, éléments traités et coût estimé. Le
coût ne doit être calculé que depuis une tarification configurable, jamais avec
des prix codés en dur.

## 10. Ajouter des citations vidéo horodatées

Étendre les chunks avec `start_time` et `end_time` :

- extraire les bornes depuis les transcripts timecodés ;
- propager les bornes des enfants vers les chunks `section` et `global` ;
- ajouter une migration PostgreSQL rétrocompatible ;
- publier et relire les champs dans toutes les requêtes de retrieval ;
- construire une URL YouTube avec `t=<secondes>s` ;
- utiliser cette URL dans le carrousel de sources.

Prévoir un backfill idempotent. Les chunks sans timecode doivent rester valides
et conserver leur URL vidéo normale.

## 11. Renforcer Phoenix, les golden cases et la CI

Créer un dataset Phoenix versionné contenant des conversations complètes et non
des questions isolées. Chaque cas doit définir au minimum :

- action attendue ;
- route attendue ;
- nombre minimal de vidéos sources ;
- contrainte de couverture ;
- latence maximale ;
- faits interdits ou claims attendus lorsque pertinent.

Évaluateurs minimaux : réponse non vide, action, couverture documentaire,
validité du juge, support des claims, couverture du fanout et quality gate
agrégé. Le quality gate doit reconnaître comme valide un fallback réellement
reconstruit à partir des seuls claims vérifiés.

Ajouter deux niveaux de CI :

- tests déterministes à chaque modification ;
- expérience live optionnelle sur runner disposant de PostgreSQL, Phoenix et
  des secrets fournisseurs.

Seuils initiaux proposés pour le live : zéro erreur d'exécution, zéro réponse
non vérifiée, au moins 90 % de succès golden et p95 inférieur à 60 secondes pour
les comparaisons. Les seuils doivent ensuite être resserrés à partir des mesures.

## 12. Ordre d'implémentation recommandé

1. Capturer la référence et écrire les tests de régression.
2. Corriger la mémoire, la reformulation et la résolution canonique.
3. Corriger les contrats de routage SQL/RAG/multi-source.
4. Mettre en place la recherche hybride hiérarchique.
5. Ajouter la matrice vidéo × axe et les registres de preuve.
6. Ajouter le vérificateur de claims et le fail-closed.
7. Centraliser retries, budget global et circuit breaker.
8. Batcher, mettre en cache et paralléliser les appels coûteux.
9. Ajouter les timecodes et effectuer le backfill.
10. Construire le dataset golden réel et les quality gates CI.
11. Rejouer la trace de référence, puis exécuter toute la suite de tests.

Chaque étape doit rester dans un commit autonome et réversible. Ne pas mélanger
une migration de données, une modification de prompt et un changement de
retrieval dans le même commit.

## 13. Critères de fin

Le chantier est terminé seulement si :

- la trace de référence est rejouée sans confusion d'entités ;
- toutes les vidéos demandées sont retrouvées par leurs identifiants canoniques ;
- la matrice couvre chaque couple vidéo × axe attendu ;
- chaque claim commun indique correctement sa portée documentaire ;
- les citations horodatées fonctionnent quand les timecodes existent ;
- un 429 simulé respecte le budget de retry et termine sans réponse non vérifiée ;
- les conversations longues restent bornées et passent les golden cases ;
- les tests unitaires, d'intégration et l'expérience Phoenix ne présentent
  aucune nouvelle régression.

## 14. Ce qui reste volontairement différé

Ne pas introduire GraphRAG au début de ce chantier. Le corpus est principalement
centré sur des vidéos et des personnes déjà représentables par PostgreSQL et une
recherche hybride. Réévaluer GraphRAG seulement si des golden cases démontrent
un besoin récurrent de raisonnement multi-sauts entre de nombreuses entités et
si la matrice vidéo × axe ne suffit pas.
