# Graphe du pipeline de réponse

Accessible depuis la FAQ à `/pipeline_reponse/`. Tous les nœuds et liens restent
visibles et fixes. Le seul contrôle est le choix d’une question d’exemple : son
parcours s’allume et les autres branches restent grisées. Aucun choix de routage
ni bouton d’étape n’est présent dans le graphe.

Les neuf premières questions reprennent exactement les libellés et l’ordre des suggestions
présentes dans `interface/index.html`. Les routes ont été vérifiées dans les
traces existantes du projet Phoenix `rag-ionis`, du 13 au 16 septembre 2026.
Ces neuf scénarios conservent leur `traceId` et leur date `observedAt` dans `app.js`.
Le panneau latéral affiche la date d’observation. Sélectionner une question ne
relance pas le RAG. Les branches non utilisées restent visibles dans le graphe.

Sources de vérité : `interface/backend/orchestration_graph.py`, `retrieval.py`,
`generation.py` et `api.py`. La sélection hybride regroupe BM25, pgvector, RRF,
le reranking optionnel et l’expansion hiérarchique. Le graphe distingue la génération
LLM de la restitution des documents ou des réponses sociales déjà prêtes.
La finalisation et la persistance sont regroupées en un nœud pédagogique.

Styles autonomes ; aucune dépendance au simulateur de préparation vidéo.

La question supplémentaire « Qui est Laura Tyan ? » illustre `annex_direct`.
Son parcours est vérifié avec la règle actuelle `select_after_plan` et la fiche
locale de Laura Tyan, sans prétendre provenir d’une nouvelle trace Phoenix.
