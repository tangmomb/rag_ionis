"use strict";

// Documentary graph grounded in orchestration_graph.py, retrieval.py and api.py.
// Nodes stay in place: selecting a question changes only the highlighted path.
const nodes = [
  {
    id: "start",
    label: "Question",
    x: 580,
    y: 12,
    w: 130,
    h: 30,
    terminal: true,
  },
  {
    id: "initialize",
    label: "Initialisation",
    sub: "Modèles et état de la requête",
    x: 580,
    y: 72,
  },
  {
    id: "reformulate",
    label: "Reformulation",
    sub: "Question + mémoire de conversation",
    x: 580,
    y: 143,
  },
  {
    id: "plan",
    label: "Planification",
    sub: "Intention, entités et recherche",
    x: 580,
    y: 214,
  },
  {
    id: "resolve_entities",
    label: "Résolution des entités",
    sub: "Personnes, entreprises, titres",
    x: 580,
    y: 298,
  },
  {
    id: "build_execution_plan",
    label: "Plan d’exécution final",
    sub: "Routage",
    x: 580,
    y: 378,
  },
  {
    id: "annex_direct",
    label: "Annexes locales",
    sub: "Connaissances ciblées",
    x: 120,
    y: 474,
  },
  {
    id: "person_clarification",
    label: "Clarification",
    sub: "Personne ambiguë",
    x: 350,
    y: 474,
  },
  {
    id: "direct",
    label: "Réponse directe",
    sub: "Échange social",
    x: 580,
    y: 474,
  },
  {
    id: "sql_search",
    label: "Recherche SQL",
    sub: "Données structurées / vidéos ciblées",
    x: 810,
    y: 474,
  },
  {
    id: "vector_search",
    label: "Recherche hybride",
    sub: "Préfiltres et embedding",
    x: 1040,
    y: 474,
  },
  {
    id: "analytics",
    label: "Text-to-SQL",
    sub: "Générer, valider, exécuter",
    x: 700,
    y: 574,
    w: 180,
  },
  {
    id: "lookup",
    label: "Recherche ciblée",
    sub: "Entités / description / transcript",
    x: 905,
    y: 574,
    w: 200,
  },
  {
    id: "retrieval",
    label: "Sélection des passages",
    sub: "BM25 → pgvector → RRF",
    extra: "Cohere (si activé) → contexte",
    x: 1040,
    y: 650,
    w: 220,
  },
  {
    id: "generate",
    label: "Génération par LLM",
    sub: "Réponse ou clarification demandée · aperçu streamé en direct",
    x: 350,
    y: 710,
    w: 210,
  },
  {
    id: "format",
    label: "Restitution sans LLM",
    sub: "Document ou réponse déjà prête",
    x: 700,
    y: 710,
    w: 220,
  },
  {
    id: "persist",
    label: "Envoi de la réponse, puis mise à jour de la mémoire",
    sub: "Réponse finale enregistrée · sources et sujets",
    badge: "Réponse à l’utilisateur",
    x: 580,
    y: 824,
    w: 300,
    highlight: true,
  },
  { id: "end", label: "Fin", x: 580, y: 916, w: 130, h: 30, terminal: true },
  {
    id: "agent",
    label: "Recherche élargie (proposée)",
    sub: "Si abstention du pipeline principal, propose à l’utilisateur d'élargir la recherche via l'agent",
    extra:
      "L'agent peut à son tour chercher dans les vidéos, les statistiques SQL, et le web en dernier recours",
    x: 915,
    y: 880,
    w: 380,
  },
].map((n) => ({ w: 208, h: 48, ...n }));
const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
const edges = [
  ["start", "initialize"],
  ["initialize", "reformulate"],
  ["reformulate", "plan"],
  ["plan", "resolve_entities"],
  ["plan", "annex_direct"],
  ["resolve_entities", "build_execution_plan"],
  ...["person_clarification", "direct", "sql_search", "vector_search"].map(
    (id) => ["build_execution_plan", id],
  ),
  ["sql_search", "analytics"],
  ["sql_search", "lookup"],
  ["vector_search", "retrieval"],
  ["annex_direct", "generate"],
  ["person_clarification", "generate"],
  ["analytics", "generate"],
  ["lookup", "generate"],
  ["retrieval", "generate"],
  ["lookup", "format"],
  ["direct", "format"],
  ["generate", "persist"],
  ["format", "persist"],
  ["persist", "end"],
  // api.py: "Élargir la recherche ?" is shown as soon as persist sends the
  // abstained answer, so the agent may run while the memory is still updating.
  ["persist", "agent"],
];
// Steps reached from an earlier step than the one before them in a path.
const attachedFrom = { agent: "persist" };
// Questions reprises de la page d’accueil, dans le même ordre.
// Routes vérifiées sur les traces Phoenix référencées dans chaque scénario.
const scenarios = {
  aymerich: {
    question: "Qui est Aymerich ?",
    title: "Identifier Aymerich",
    route: "sql_search",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "resolve_entities",
      "build_execution_plan",
      "sql_search",
      "lookup",
      "generate",
      "persist",
      "end",
    ],
    explanation:
      "La résolution du nom conduit à une recherche ciblée en SQL pour retrouver les vidéos qui parlent d’Aymerich.",
    steps: [
      "Le planner extrait les personnes mentionnées dans la question.",
      "La résolution rapproche les noms des intervenants et des mentions dans les transcripts.",
      "SQL retrouve les vidéos ciblées ; les résultats sont dédupliqués.",
      "Le LLM prépare la réponse à partir des sources, puis la conversation est mémorisée.",
    ],
    note: "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    traceId: "07c2c51ffc765ad3e217cf08673620b2",
    observedAt: "2026-09-16T16:53:35.505646+00:00",
  },
  salim: {
    question: "J'ai besoin de la transcription de la vidéo de Salim.",
    title: "Retrouver la transcription de Salim",
    route: "sql_search",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "resolve_entities",
      "build_execution_plan",
      "sql_search",
      "lookup",
      "format",
      "persist",
      "end",
    ],
    explanation:
      "La demande de transcription est traitée en SQL. Le document est ensuite restitué sans LLM de réponse.",
    steps: [
      "Le planner identifie la personne et le document demandé.",
      "La résolution rapproche le nom des données connues, puis SQL retrouve les vidéos candidates.",
      "La transcription est récupérée et mise en forme sans LLM de réponse.",
      "Le document restitué et les sources rejoignent la conversation.",
    ],
    note: "Parcours observé dans Phoenix le 14/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    traceId: "e3d3e8ac0499bb2205228019a5c787f8",
    observedAt: "2026-09-14T19:26:23.355828+00:00",
  },
  views: {
    question: "Quelle vidéo a le plus de vues ?",
    title: "Trouver la vidéo la plus vue",
    route: "sql_search",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "resolve_entities",
      "build_execution_plan",
      "sql_search",
      "analytics",
      "generate",
      "persist",
      "end",
    ],
    explanation:
      "La question porte sur une statistique des vidéos : le plan sélectionne la route analytique Text-to-SQL.",
    steps: [
      "Le planner reconnaît une demande analytique.",
      "Le LLM Text-to-SQL produit une requête qui est validée puis exécutée.",
      "Les résultats structurés servent à générer la réponse.",
      "La conversation est mémorisée ; aucun embedding ni reranking n’intervient.",
    ],
    note: "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    traceId: "8df47c98d6a70c563c318e89e716a6f8",
    observedAt: "2026-09-16T12:40:26.331086+00:00",
  },
  views_growth: {
    question: "Quelle vidéo a gagné le plus de vues entre août et septembre ?",
    title: "Comparer les gains de vues",
    route: "sql_search",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "resolve_entities",
      "build_execution_plan",
      "sql_search",
      "analytics",
      "generate",
      "persist",
      "end",
    ],
    explanation:
      "La question demande une évolution du nombre de vues entre deux mois : la route analytique interroge l’historique des statistiques.",
    steps: [
      "Le planner reconnaît une demande analytique.",
      "Le LLM Text-to-SQL produit une requête qui est validée puis exécutée.",
      "Les résultats structurés servent à générer la réponse.",
      "La conversation est mémorisée ; aucun embedding ni reranking n’intervient.",
    ],
    note: "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    traceId: "6fffb70aba10d98d42889d9812154451",
    observedAt: "2026-09-16T12:40:35.210342+00:00",
  },
  gaelle: {
    question: "J'ai besoin de la description de la vidéo de Gaëlle.",
    title: "Retrouver la description de Gaëlle",
    route: "sql_search",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "resolve_entities",
      "build_execution_plan",
      "sql_search",
      "lookup",
      "format",
      "persist",
      "end",
    ],
    explanation:
      "La demande de description suit la route documentaire SQL, puis une restitution sans LLM de réponse.",
    steps: [
      "Le planner identifie la personne et le document demandé.",
      "La résolution rapproche le nom des données connues, puis SQL retrouve les vidéos candidates.",
      "La description est récupérée et mise en forme sans LLM de réponse.",
      "Le document restitué et les sources rejoignent la conversation.",
    ],
    note: "Parcours observé dans Phoenix le 14/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    traceId: "0f0555ade4537a426fd939682599e11b",
    observedAt: "2026-09-14T21:22:18.920476+00:00",
  },
  comparison: {
    question: "Des points communs entre Sophie Ollivier et Chloé Leprètre ?",
    title: "Comparer deux parcours",
    route: "sql_search",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "resolve_entities",
      "build_execution_plan",
      "sql_search",
      "lookup",
      "generate",
      "persist",
      "end",
    ],
    explanation:
      "Les deux personnes identifiées dirigent la recherche vers SQL. La comparaison est ensuite rédigée à partir des vidéos retrouvées.",
    steps: [
      "Le planner extrait les personnes mentionnées dans la question.",
      "La résolution rapproche les noms des intervenants et des mentions dans les transcripts.",
      "SQL retrouve les vidéos ciblées ; les résultats sont dédupliqués.",
      "Le LLM prépare la réponse à partir des sources, puis la conversation est mémorisée.",
    ],
    note: "Parcours observé dans Phoenix le 13/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    traceId: "adfb6de489c57912b143668eeb7d0726",
    observedAt: "2026-09-13T20:25:20.964642+00:00",
  },
  ionis: {
    question: "Pourquoi faire Ionis-STM ?",
    title: "Explorer les raisons de choisir Ionis-STM",
    route: "vector_search",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "resolve_entities",
      "build_execution_plan",
      "vector_search",
      "retrieval",
      "generate",
      "persist",
      "end",
    ],
    explanation:
      "Cette question générale sur la formation emprunte la recherche hybride pour retrouver des passages pertinents.",
    steps: [
      "Le reformulateur prépare la question, puis le planner définit la recherche.",
      "BM25 et pgvector retrouvent les passages ; RRF fusionne leurs classements.",
      "Le reranking Cohere est activé dans cette trace, puis le contexte hiérarchique est enrichi.",
      "Le LLM génère une réponse à partir des sources et la conversation est mémorisée.",
    ],
    note: "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    traceId: "765c4241010d0a950f5ad46680affd1c",
    observedAt: "2026-09-16T12:40:45.857312+00:00",
  },
  ocean: {
    question: "C'est quoi l'océan rouge ?",
    title: "Expliquer l’océan rouge",
    route: "vector_search",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "resolve_entities",
      "build_execution_plan",
      "vector_search",
      "retrieval",
      "generate",
      "persist",
      "end",
    ],
    explanation:
      "La question porte sur une notion abordée dans les vidéos, sans personne ni document ciblé : la recherche est hybride.",
    steps: [
      "Le reformulateur prépare la question, puis le planner définit la recherche.",
      "BM25 et pgvector retrouvent les passages ; RRF fusionne leurs classements.",
      "Le reranking Cohere est activé dans cette trace, puis le contexte hiérarchique est enrichi.",
      "Le LLM génère une réponse à partir des sources et la conversation est mémorisée.",
    ],
    note: "Parcours observé dans Phoenix le 13/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    traceId: "7c3c473acacdfd6f07b65a3dbd36097a",
    observedAt: "2026-09-13T19:27:52.943945+00:00",
  },
  fadila: {
    question: "Quelles questions ont été posées à Fadila ?",
    title: "Retrouver les questions posées à Fadila",
    route: "sql_search",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "resolve_entities",
      "build_execution_plan",
      "sql_search",
      "lookup",
      "generate",
      "persist",
      "end",
    ],
    explanation:
      "La personne identifiée permet de retrouver les vidéos correspondantes en SQL, puis d’en extraire les questions dans la réponse.",
    steps: [
      "Le planner extrait les personnes mentionnées dans la question.",
      "La résolution rapproche les noms des intervenants et des mentions dans les transcripts.",
      "SQL retrouve les vidéos ciblées ; les résultats sont dédupliqués.",
      "Le LLM prépare la réponse à partir des sources, puis la conversation est mémorisée.",
    ],
    note: "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    traceId: "2b8db3209a77e1aaeed15c224f42517d",
    observedAt: "2026-09-16T16:54:15.371108+00:00",
  },
  laura: {
    question: "Qui est Laura Tyan ?",
    title: "Utiliser la fiche de Laura Tyan",
    route: "annex_direct",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "annex_direct",
      "generate",
      "persist",
      "end",
    ],
    explanation:
      "Laura Tyan figure dans les connaissances annexes. Dès que le planner identifie son nom, le pipeline utilise sa fiche locale.",
    steps: [
      "La question est reformulée, puis le planner identifie Laura Tyan.",
      "Sa présence dans l’annexe déclenche la route annex_direct, avant la résolution des entités en base.",
      "Le LLM prépare la réponse à partir des connaissances locales ciblées, sans recherche SQL ni vectorielle.",
      "La réponse est finalisée et la conversation est mémorisée.",
    ],
    note: "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cette trace enregistrée ; elle ne relance pas le RAG.",
    traceId: "d015c3afa2aeb3550edb5bb5d5fc7fe6",
    observedAt: "2026-09-16T21:55:49.298987+00:00",
  },
  australie: {
    question: "Quelle est la capitale de l'Australie ?",
    title: "S’abstenir, puis élargir la recherche",
    route: "vector_search",
    path: [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "resolve_entities",
      "build_execution_plan",
      "vector_search",
      "retrieval",
      "generate",
      "persist",
      "end",
      "agent",
    ],
    explanation:
      "Aucune vidéo ne parle de la capitale de l’Australie. Plutôt que d’inventer, le RAG répond « À première vue, je n’ai pas la réponse dans les vidéos » et propose d’élargir la recherche. L’agent trouve alors la réponse sur le web.",
    steps: [
      "La recherche hybride ne trouve aucun passage qui réponde à la question.",
      "Le LLM s’abstient ; son constat (ce que les sources couvrent ou non) est conservé pour l’agent.",
      "La réponse affiche « À première vue, je n’ai pas la réponse dans les vidéos » et le bouton « Élargir la recherche ? ».",
      "Au clic, l’agent choisit ses outils. Il n’a rien cherché dans la base ici ; invité à tenter le web avant de renoncer, il y trouve « Canberra », avec ses sources.",
    ],
    note: "Parcours observé dans Phoenix le 13/10/2026 : trace RAG bbdb64bb…, puis trace de l’agent 52c24a10…, lancée au clic. La sélection affiche ces traces enregistrées ; elle ne relance pas le RAG.",
    traceId: "bbdb64bbfb35607b8e88d38f6269ca19",
    observedAt: "2026-10-12T22:50:55.650593+00:00",
  },
};
// Sorties LLM enregistrées avec les scénarios : le visualiseur reste autonome
// même si les traces Phoenix sont purgées ou indisponibles.
const recordedLlmOutputs = {
  aymerich: {
    reformulate: {
      follow_up: false,
      reformulated_question: "Qui est Aymerich ?",
      topic: "Aymerich",
    },
    plan: {
      route: "search",
      analytics: false,
      query_text: "Qui est Aymerich ?",
      query_text_bm25: "Aymerich identité",
      title_hints: [],
      persons: ["Aymerich"],
      companies: [],
      published_after: null,
      published_before: null,
    },
  },
  salim: {
    reformulate: {
      follow_up: false,
      reformulated_question:
        "Pouvez-vous fournir la transcription de la vidéo de Salim ?",
      topic: "transcription de la vidéo de Salim",
    },
    plan: {
      route: "search",
      analytics: false,
      query_text: "Quelle est la transcription de la vidéo de Salim ?",
      query_text_bm25: "transcription vidéo Salim",
      title_hints: [],
      persons: ["Salim"],
      companies: [],
      published_after: null,
      published_before: null,
    },
  },
  views: {
    reformulate: {
      follow_up: false,
      reformulated_question: "Quelle vidéo a le plus de vues ?",
      topic: "vidéos",
    },
    plan: {
      route: "search",
      analytics: true,
      query_text: "Quelle vidéo a le plus de vues ?",
      query_text_bm25: "vidéo plus vues",
      title_hints: [],
      persons: [],
      companies: [],
      published_after: null,
      published_before: null,
    },
  },
  views_growth: {
    reformulate: {
      follow_up: false,
      reformulated_question:
        "Quelle vidéo a gagné le plus de vues entre août et septembre ?",
      topic: "vidéos et vues entre août et septembre",
    },
    plan: {
      route: "search",
      analytics: true,
      query_text:
        "Quelle vidéo a gagné le plus de vues entre août et septembre ?",
      query_text_bm25: "vidéo plus vues août septembre",
      title_hints: [],
      persons: [],
      companies: [],
      published_after: null,
      published_before: null,
    },
  },
  gaelle: {
    reformulate: {
      follow_up: false,
      reformulated_question:
        "Quelle est la description de la vidéo de Gaëlle ?",
      topic: "description de la vidéo de Gaëlle",
    },
    plan: {
      route: "search",
      analytics: false,
      query_text: "Quelle est la description de la vidéo de Gaëlle ?",
      query_text_bm25: "description vidéo Gaëlle",
      title_hints: [],
      persons: ["Gaëlle"],
      companies: [],
      published_after: null,
      published_before: null,
    },
  },
  comparison: {
    reformulate: {
      follow_up: false,
      reformulated_question:
        "Quels sont les points communs entre Sophie Ollivier et Chloé Leprètre ?",
      topic: "points communs entre Sophie Ollivier et Chloé Leprètre",
    },
    plan: {
      route: "search",
      analytics: false,
      query_text:
        "Quels sont les points communs entre Sophie Ollivier et Chloé Leprètre ?",
      query_text_bm25: "points communs Sophie Ollivier Chloé Leprètre",
      title_hints: [],
      persons: ["Sophie Ollivier", "Chloé Leprètre"],
      companies: [],
      published_after: null,
      published_before: null,
    },
  },
  ionis: {
    reformulate: {
      follow_up: false,
      reformulated_question:
        "Pourquoi choisir de faire des études à Ionis-STM ?",
      topic: "choix des études à Ionis-STM",
    },
    plan: {
      route: "search",
      analytics: false,
      query_text: "Pourquoi choisir de faire des études à Ionis-STM ?",
      query_text_bm25: "études Ionis-STM raisons choix",
      title_hints: [],
      persons: [],
      companies: ["Ionis-STM"],
      published_after: null,
      published_before: null,
    },
  },
  ocean: {
    reformulate: {
      follow_up: false,
      reformulated_question: "Qu'est-ce que l'océan rouge ?",
      topic: "océan rouge",
    },
    plan: {
      route: "search",
      analytics: false,
      analytics_scope: null,
      analytics_metric: null,
      analytics_order: null,
      analytics_rank_start: null,
      analytics_rank_end: null,
      query_text: "Qu'est-ce que l'océan rouge ?",
      query_text_bm25: "océan rouge définition",
      title_hints: [],
      persons: [],
      companies: [],
      published_after: null,
      published_before: null,
    },
  },
  fadila: {
    reformulate: {
      follow_up: false,
      reformulated_question: "Quelles questions ont été posées à Fadila ?",
      topic: "questions posées à Fadila",
    },
    plan: {
      route: "search",
      analytics: false,
      query_text: "Quelles questions ont été posées à Fadila ?",
      query_text_bm25: "questions posées Fadila",
      title_hints: [],
      persons: ["Fadila"],
      companies: [],
      published_after: null,
      published_before: null,
    },
  },
  laura: {
    reformulate: {
      follow_up: false,
      reformulated_question: "Qui est Laura Tyan ?",
      topic: "Laura Tyan",
    },
    plan: {
      route: "search",
      analytics: false,
      query_text: "Qui est Laura Tyan ?",
      query_text_bm25: "Laura Tyan identité biographie",
      title_hints: [],
      persons: ["Laura Tyan"],
      companies: [],
      published_after: null,
      published_before: null,
    },
  },
  australie: {
    reformulate: {
      follow_up: false,
      reformulated_question: "Quelle est la capitale de l'Australie ?",
      topic: "capitale de l'Australie",
    },
    plan: {
      route: "search",
      analytics: false,
      query_text: "Quelle est la capitale de l'Australie ?",
      query_text_bm25: "capitale Australie",
      title_hints: [],
      persons: [],
      companies: [],
      published_after: null,
      published_before: null,
    },
    agent: {
      tool_calls: [],
      web_nudged: true,
      web_searched: true,
      answer:
        "D'après le web, la capitale officielle de l'Australie est **Canberra**.",
      web_sources: [
        "https://fr.wikipedia.org/wiki/Canberra",
        "https://fr.wikipedia.org/wiki/Australie",
        "https://www.universalis.fr/encyclopedie/canberra/",
      ],
    },
  },
};
const staticStepOutputs = {
  initialize: "La requête, la conversation et les modèles sont préparés.",
  resolve_entities:
    "Les personnes, entreprises et titres extraits sont rapprochés des données connues.",
  build_execution_plan:
    "Le plan du LLM est complété par les règles de routage déterministes.",
  annex_direct:
    "La fiche locale correspondante est ajoutée au contexte de réponse.",
  person_clarification:
    "Une demande de précision est préparée pour identifier la personne visée.",
  direct:
    "La réponse sociale déjà prête est sélectionnée, sans recherche documentaire.",
  sql_search:
    "Les données structurées et les vidéos candidates sont recherchées en SQL.",
  vector_search:
    "La recherche hybride prépare les requêtes BM25 et vectorielle.",
  analytics:
    "La requête Text-to-SQL est validée puis exécutée avec des droits de lecture seuls.",
  lookup:
    "Les documents, descriptions ou transcriptions demandés sont récupérés.",
  retrieval:
    "Les passages sont classés, fusionnés puis enrichis de leur contexte hiérarchique.",
  generate:
    "Le LLM produit la réponse à partir du contexte et des sources retenues.",
  format:
    "La réponse ou le document déjà prêt est mis en forme sans appel LLM.",
  persist:
    "La réponse est finalisée (sources du carrousel, abstention éventuelle), enregistrée puis envoyée à l’utilisateur. La mémoire de conversation est mise à jour ensuite, sans retarder l’affichage : le sujet de l’échange et les vidéos citées, avec un résumé de l’ancien sujet quand la conversation change de sujet.",
  end: "Fin du traitement de la question.",
  agent:
    "Étape facultative, jamais automatique : après une abstention du pipeline principal, la réponse propose le bouton « Élargir la recherche ? », et l’agent ne démarre que si l’utilisateur clique. Il cherche alors dans les vidéos, les statistiques en SQL validé, et le web en dernier recours. Après une recherche vectorielle, il part de zéro (sans relancer la même recherche) ; pour les autres routes, une première recherche dans les vidéos sur la question reformulée lui est fournie au départ. Sa réponse s’affiche à part, avec ses sources. Le bouton apparaît dès l’envoi de la réponse : l’agent peut donc tourner pendant la mise à jour de la mémoire, et sa réponse n’y est pas ajoutée.",
};
const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const edgeId = (from, to) => `${from}--${to}`;
function edgePath(from, to) {
  const a = byId[from],
    b = byId[to];
  const sy = a.y + a.h,
    ey = b.y;
  const between = (t) => sy + (ey - sy) * t;
  // Route the long shortcuts outside the intermediate nodes. Every point is
  // derived from the laid-out nodes, so spacing changes keep the routes.
  if (from === "plan" && to === "annex_direct")
    return `M ${a.x} ${sy} C ${a.x} ${sy + 45}, ${b.x} ${sy + 45}, ${b.x} ${ey}`;
  if (from === "vector_search")
    return `M ${a.x} ${sy} C ${a.x + 130} ${sy + 30}, ${b.x + 130} ${ey - 30}, ${b.x} ${ey}`;
  if (from === "retrieval") {
    const [x1, y1] = [a.x - a.w / 2, a.y + a.h / 2];
    const [x2, y2] = [b.x + b.w / 2, b.y + b.h / 2];
    return `M ${x1} ${y1} C ${x1 - 140} ${y1}, ${x2 + 160} ${y2 - 70}, ${x2} ${y2}`;
  }
  if (from === "lookup" && to === "generate")
    return `M ${a.x} ${sy} C ${a.x} ${between(0.35)}, ${b.x + 80} ${between(0.35)}, ${b.x + 80} ${ey}`;
  if (to === "agent") {
    const y = a.y + a.h / 2;
    return `M ${a.x + a.w / 2} ${y} L ${b.x - b.w / 2} ${y}`;
  }
  return `M ${a.x} ${sy} C ${a.x} ${between(0.5)}, ${b.x} ${between(0.5)}, ${b.x} ${ey}`;
}
const svg = document.querySelector("#pipeline-graph");
// Layout. The node coordinates above sit on a compact grid: spread them so
// the graph breathes (pan and zoom below handle the larger canvas), then size
// every node to its measured text.
const SPREAD_X = 1.35;
const SPREAD_Y = 1.6;
const MAX_LINE_WIDTH = 330;
const measureContext = document.createElement("canvas").getContext("2d");
const fontFamily = getComputedStyle(document.documentElement).fontFamily;
function textWidth(text, size, weight) {
  measureContext.font = `${weight} ${size}px ${fontFamily}`;
  return measureContext.measureText(text).width;
}
// Only text wider than MAX_LINE_WIDTH wraps; nodes widen for the rest.
function wrapText(text) {
  const lines = [];
  let line = "";
  for (const word of text.split(" ")) {
    const candidate = line ? `${line} ${word}` : word;
    if (line && textWidth(candidate, 10, 400) > MAX_LINE_WIDTH) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  }
  if (line) lines.push(line);
  return lines;
}
nodes.forEach((n) => {
  n.x = Math.round(n.x * SPREAD_X);
  n.y = Math.round(n.y * SPREAD_Y);
  n.subLines = n.sub ? wrapText(n.sub) : [];
  n.extraLines = n.extra ? wrapText(n.extra) : [];
  const widest = Math.max(
    textWidth(n.label, 14, 500),
    ...[...n.subLines, ...n.extraLines].map((line) => textWidth(line, 10, 400)),
    n.badge ? textWidth(n.badge, 9, 500) + 28 : 0,
  );
  n.w = Math.max(n.w, Math.ceil(widest) + 40);
  if (n.terminal) return;
  // Baselines: title, then one 14px step per line, a gap before the extra
  // block and the badge, and bottom padding.
  let y = 23;
  n.lineYs = [];
  for (const _ of n.subLines) n.lineYs.push((y += 15));
  if (n.extraLines.length) y += 4;
  for (const _ of n.extraLines) n.lineYs.push((y += 15));
  if (n.badge) {
    n.badgeY = y + 9;
    y = n.badgeY + 16;
  }
  n.h = Math.max(n.h, y + 13);
});
// The proposed expanded search sits on the right of the step that sends the
// answer, centred on it.
{
  const persist = byId.persist;
  const agent = byId.agent;
  agent.x = persist.x + persist.w / 2 + 90 + agent.w / 2;
  agent.y = persist.y + persist.h / 2 - agent.h / 2;
}
function nodeTextMarkup(n) {
  if (n.terminal) {
    return `<text x="${n.w / 2}" y="${n.h / 2 + 5}">${escapeHtml(n.label)}</text>`;
  }
  const lines = [...n.subLines, ...n.extraLines].map(
    (line, index) =>
      `<text class="subtitle" x="${n.w / 2}" y="${n.lineYs[index]}">${escapeHtml(line)}</text>`,
  );
  const badge = n.badge
    ? `<g class="badge"><rect x="${n.w / 2 - (textWidth(n.badge, 9, 500) + 24) / 2}" y="${n.badgeY}" width="${textWidth(n.badge, 9, 500) + 24}" height="16" rx="8"/><text x="${n.w / 2}" y="${n.badgeY + 11.5}">${escapeHtml(n.badge)}</text></g>`
    : "";
  return `<text x="${n.w / 2}" y="23">${escapeHtml(n.label)}</text>${lines.join("")}${badge}`;
}
const bounds = nodes.reduce(
  (box, n) => ({
    minX: Math.min(box.minX, n.x - n.w / 2),
    minY: Math.min(box.minY, n.y),
    maxX: Math.max(box.maxX, n.x + n.w / 2),
    maxY: Math.max(box.maxY, n.y + n.h),
  }),
  { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity },
);
const fullView = {
  x: bounds.minX - 60,
  y: bounds.minY - 30,
  w: bounds.maxX - bounds.minX + 120,
  h: bounds.maxY - bounds.minY + 60,
};
svg.innerHTML = `<title id="graph-title">Graphe complet du pipeline de réponse RAG</title><desc id="graph-description"></desc>
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="#383838" stroke-width="1.3"/></marker><marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="#ffffff" stroke-width="1.6"/></marker></defs>
  <g class="edges">${edges.map(([from, to]) => `<path id="${edgeId(from, to)}" class="edge${to === "agent" ? " optional" : ""}" d="${edgePath(from, to)}" marker-end="url(#arrow)"/>`).join("")}</g>
  <g class="nodes">${nodes.map((n) => `<g id="node-${n.id}" class="node ${n.terminal ? "terminal" : ""} ${n.highlight ? "highlight" : ""}" transform="translate(${n.x - n.w / 2},${n.y})"><rect width="${n.w}" height="${n.h}" rx="7"/>${nodeTextMarkup(n)}</g>`).join("")}</g>`;
// Pan and zoom, as in LangSmith Studio: the wheel zooms around the pointer,
// dragging pans, and the buttons zoom or fit the whole graph back.
let view = { ...fullView };
function applyView() {
  svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`);
}
function svgPoint(clientX, clientY) {
  const point = new DOMPoint(clientX, clientY).matrixTransform(svg.getScreenCTM().inverse());
  return [point.x, point.y];
}
function zoomAt(clientX, clientY, factor) {
  const [px, py] = svgPoint(clientX, clientY);
  const w = Math.min(Math.max(view.w * factor, fullView.w * 0.12), fullView.w * 1.6);
  const scale = w / view.w;
  view = { x: px - (px - view.x) * scale, y: py - (py - view.y) * scale, w, h: view.h * scale };
  applyView();
  hideTooltip();
}
function zoomFromCenter(factor) {
  const box = svg.getBoundingClientRect();
  zoomAt(box.left + box.width / 2, box.top + box.height / 2, factor);
}
svg.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    zoomAt(event.clientX, event.clientY, event.deltaY < 0 ? 0.88 : 1 / 0.88);
  },
  { passive: false },
);
let drag = null;
svg.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  drag = { x: event.clientX, y: event.clientY, view: { ...view }, moved: false };
  svg.setPointerCapture(event.pointerId);
});
svg.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const dx = event.clientX - drag.x;
  const dy = event.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 4) return;
  if (!drag.moved) {
    drag.moved = true;
    svg.classList.add("panning");
    hideTooltip();
  }
  // Screen pixels to viewBox units, from the current on-screen scale.
  const unit = 1 / svg.getScreenCTM().a;
  view = { ...drag.view, x: drag.view.x - dx * unit, y: drag.view.y - dy * unit };
  applyView();
});
function endDrag() {
  drag = null;
  svg.classList.remove("panning");
}
svg.addEventListener("pointerup", endDrag);
svg.addEventListener("pointercancel", endDrag);
const zoomControls = document.createElement("div");
zoomControls.className = "graph-zoom";
zoomControls.innerHTML = `<button type="button" data-zoom="in" aria-label="Zoomer">+</button><button type="button" data-zoom="out" aria-label="Dézoomer">−</button><button type="button" data-zoom="fit" aria-label="Afficher tout le graphe">⤢</button>`;
zoomControls.addEventListener("click", (event) => {
  const action = event.target.closest("button")?.dataset.zoom;
  if (action === "in") zoomFromCenter(0.8);
  if (action === "out") zoomFromCenter(1 / 0.8);
  if (action === "fit") {
    view = { ...fullView };
    applyView();
  }
});
svg.parentElement.append(zoomControls);
applyView();
const select = document.querySelector("#scenario");
select.innerHTML = Object.entries(scenarios)
  .map(([id, s]) => `<option value="${id}">${escapeHtml(s.question)}</option>`)
  .join("");

function render() {
  const s = scenarios[select.value];
  const activeNodes = new Set(s.path);
  const activeEdges = new Set(
    s.path
      .slice(1)
      .map((id, index) => edgeId(attachedFrom[id] ?? s.path[index], id)),
  );
  nodes.forEach((n) =>
    document
      .getElementById(`node-${n.id}`)
      .classList.toggle("active", activeNodes.has(n.id)),
  );
  edges.forEach(([from, to]) => {
    const line = document.getElementById(edgeId(from, to));
    const active = activeEdges.has(line.id);
    line.classList.toggle("active", active);
    line.setAttribute(
      "marker-end",
      `url(#${active ? "arrow-active" : "arrow"})`,
    );
  });
  // Paint lit edges last so crossing inactive branches cannot obscure the route.
  svg
    .querySelectorAll(".edge.active")
    .forEach((line) => line.parentNode.append(line));
  document.querySelector("#graph-description").textContent =
    `Question : ${s.question}. Parcours éclairé : ${s.path.map((id) => byId[id].label).join(" → ")}. Toutes les autres branches restent visibles en grisé.`;
  document.querySelector("#scenario-details").innerHTML =
    `<span class="eyebrow">Parcours de la question</span><h2>${escapeHtml(s.title)}</h2><span class="route-tag">${s.route}</span><p>${escapeHtml(s.explanation)}</p><h3>Ce qui se passe</h3><ol>${s.steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ol><div class="context"><h3>${s.traceId ? "Parcours observé" : "Règle vérifiée"}</h3><p>${escapeHtml(s.note)}</p></div>`;
}
select.addEventListener("change", render);
render();

const tooltip = document.createElement("div");
tooltip.id = "step-tooltip";
tooltip.className = "step-tooltip";
tooltip.setAttribute("role", "tooltip");
tooltip.hidden = true;
document.body.append(tooltip);
let tooltipNode = null;
let closeTimer;
function hideTooltip() {
  clearTimeout(closeTimer);
  tooltipNode?.removeAttribute("aria-describedby");
  tooltipNode = null;
  tooltip.hidden = true;
}

function showTooltip(node) {
  clearTimeout(closeTimer);
  tooltipNode?.removeAttribute("aria-describedby");
  tooltipNode = node;
  const id = node.id.slice(5);
  const scenario = scenarios[select.value];
  const active = scenario.path.includes(id);
  let outputMarkup = "";
  if (id === "start") {
    outputMarkup = `<p>${escapeHtml(scenario.question)}</p>`;
  } else if (!active) {
    outputMarkup =
      "<p>Aucune sortie : cette étape n'est pas empruntée par la question sélectionnée.</p>";
  } else if (
    id === "reformulate" ||
    id === "plan" ||
    (id === "agent" && recordedLlmOutputs[select.value]?.agent)
  ) {
    const output = recordedLlmOutputs[select.value]?.[id];
    outputMarkup = output
      ? `<pre>${escapeHtml(JSON.stringify(output, null, 2))}</pre>`
      : "<p>Sortie enregistrée indisponible pour cet exemple.</p>";
  } else if (
    typeof recordedStepOutputs !== "undefined" &&
    recordedStepOutputs[select.value]?.[id]
  ) {
    // step_outputs.js: the step's span output from the scenario's Phoenix trace.
    outputMarkup = `<pre>${escapeHtml(JSON.stringify(recordedStepOutputs[select.value][id], null, 2))}</pre>`;
  } else {
    outputMarkup = `<p>${escapeHtml(staticStepOutputs[id] || "Cette étape produit la suite du parcours.")}</p>`;
  }
  tooltip.innerHTML = `<strong>${escapeHtml(byId[id].label)} · Sortie</strong>${outputMarkup}`;
  tooltip.hidden = false;
  node.setAttribute("aria-describedby", tooltip.id);
  const rect = node.getBoundingClientRect();
  const box = tooltip.getBoundingClientRect();
  let left = rect.right + 12;
  if (left + box.width > innerWidth - 12) left = rect.left - box.width - 12;
  tooltip.style.left = `${Math.max(12, Math.min(left, innerWidth - box.width - 12))}px`;
  tooltip.style.top = `${Math.max(12, Math.min(rect.top, innerHeight - box.height - 12))}px`;
}
function scheduleHide() {
  closeTimer = setTimeout(hideTooltip, 140);
}
svg.querySelectorAll(".node").forEach((node) => {
  node.setAttribute("tabindex", "0");
  node.setAttribute("aria-label", byId[node.id.slice(5)].label);
  node.addEventListener("mouseenter", () => showTooltip(node));
  node.addEventListener("mouseleave", scheduleHide);
  node.addEventListener("focus", () => showTooltip(node));
  node.addEventListener("blur", hideTooltip);
});
tooltip.addEventListener("mouseenter", () => clearTimeout(closeTimer));
tooltip.addEventListener("mouseleave", scheduleHide);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideTooltip();
});
select.addEventListener("change", hideTooltip);
window.addEventListener("resize", hideTooltip);
window.addEventListener(
  "scroll",
  (event) => {
    if (!tooltip.contains(event.target)) hideTooltip();
  },
  true,
);
