"use strict";

// Documentary graph grounded in orchestration_graph.py, retrieval.py and api.py.
// Nodes stay in place: selecting a question changes only the highlighted path.
const nodes = [
  { id: "start", label: "Question", x: 580, y: 12, w: 130, h: 30, terminal: true },
  { id: "initialize", label: "Initialisation", sub: "Modèles et état de la requête", x: 580, y: 72 },
  { id: "reformulate", label: "Reformulation", sub: "Question + mémoire de conversation", x: 580, y: 143 },
  { id: "plan", label: "Planification", sub: "Intention, entités et recherche", x: 580, y: 214 },
  { id: "resolve_entities", label: "Résolution des entités", sub: "Personnes, entreprises, titres", x: 580, y: 298 },
  { id: "build_execution_plan", label: "Routage", sub: "Plan d’exécution validé", x: 580, y: 378 },
  { id: "annex_direct", label: "Annexes locales", sub: "Connaissances ciblées", x: 120, y: 474 },
  { id: "person_clarification", label: "Clarification", sub: "Personne ambiguë", x: 350, y: 474 },
  { id: "direct", label: "Réponse directe", sub: "Échange social", x: 580, y: 474 },
  { id: "sql_search", label: "Recherche SQL", sub: "Données structurées / vidéos ciblées", x: 810, y: 474 },
  { id: "vector_search", label: "Recherche hybride", sub: "Préfiltres et embedding", x: 1040, y: 474 },
  { id: "analytics", label: "Text-to-SQL", sub: "Générer, valider, exécuter", x: 700, y: 574, w: 180 },
  { id: "lookup", label: "Recherche ciblée", sub: "Entités / description / transcript", x: 905, y: 574, w: 200 },
  { id: "retrieval", label: "Sélection des passages", sub: "BM25 → pgvector → RRF", extra: "Cohere (si activé) → contexte", x: 1040, y: 650, w: 220, h: 66 },
  { id: "generate", label: "Génération par LLM", sub: "Réponse ou clarification", x: 350, y: 710, w: 210 },
  { id: "format", label: "Restitution sans LLM", sub: "Document ou réponse déjà prête", x: 700, y: 710, w: 220 },
  { id: "persist", label: "Finalisation et mémoire", sub: "Réponse, sources et conversation", x: 580, y: 794, w: 240 },
  { id: "end", label: "Réponse", x: 580, y: 874, w: 130, h: 30, terminal: true },
].map(n => ({ w: 208, h: 48, ...n }));
const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
const edges = [
  ["start", "initialize"], ["initialize", "reformulate"], ["reformulate", "plan"],
  ["plan", "resolve_entities"], ["plan", "annex_direct"],
  ["resolve_entities", "build_execution_plan"],
  ...["person_clarification", "direct", "sql_search", "vector_search"].map(id => ["build_execution_plan", id]),
  ["sql_search", "analytics"], ["sql_search", "lookup"], ["vector_search", "retrieval"],
  ["annex_direct", "generate"], ["person_clarification", "generate"],
  ["analytics", "generate"], ["lookup", "generate"], ["retrieval", "generate"],
  ["lookup", "format"], ["direct", "format"],
  ["generate", "persist"], ["format", "persist"], ["persist", "end"],
];
// Questions reprises de la page d’accueil, dans le même ordre.
// Routes vérifiées sur les traces Phoenix référencées dans chaque scénario.
const scenarios = {
  "aymerich": {
    "question": "Qui est Aymerich ?",
    "title": "Identifier Aymerich",
    "route": "sql_search",
    "path": [
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
      "end"
    ],
    "explanation": "La résolution du nom conduit à une recherche ciblée en SQL pour retrouver les vidéos qui parlent d’Aymerich.",
    "steps": [
      "Le planner extrait les personnes mentionnées dans la question.",
      "La résolution rapproche les noms des intervenants et des mentions dans les transcripts.",
      "SQL retrouve les vidéos ciblées ; les résultats sont dédupliqués.",
      "Le LLM prépare la réponse à partir des sources, puis la conversation est mémorisée."
    ],
    "note": "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    "traceId": "07c2c51ffc765ad3e217cf08673620b2",
    "observedAt": "2026-09-16T16:53:35.505646+00:00"
  },
  "salim": {
    "question": "J'ai besoin de la transcription de la vidéo de Salim.",
    "title": "Retrouver la transcription de Salim",
    "route": "sql_search",
    "path": [
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
      "end"
    ],
    "explanation": "La demande de transcription est traitée en SQL. Le document est ensuite restitué sans LLM de réponse.",
    "steps": [
      "Le planner identifie la personne et le document demandé.",
      "La résolution rapproche le nom des données connues, puis SQL retrouve les vidéos candidates.",
      "La transcription est récupérée et mise en forme sans LLM de réponse.",
      "Le document restitué et les sources rejoignent la conversation."
    ],
    "note": "Parcours observé dans Phoenix le 14/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    "traceId": "e3d3e8ac0499bb2205228019a5c787f8",
    "observedAt": "2026-09-14T19:26:23.355828+00:00"
  },
  "views": {
    "question": "Quelle vidéo a le plus de vues ?",
    "title": "Trouver la vidéo la plus vue",
    "route": "sql_search",
    "path": [
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
      "end"
    ],
    "explanation": "La question porte sur une statistique des vidéos : le plan sélectionne la route analytique Text-to-SQL.",
    "steps": [
      "Le planner reconnaît une demande analytique.",
      "Le LLM Text-to-SQL produit une requête qui est validée puis exécutée.",
      "Les résultats structurés servent à générer la réponse.",
      "La conversation est mémorisée ; aucun embedding ni reranking n’intervient."
    ],
    "note": "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    "traceId": "8df47c98d6a70c563c318e89e716a6f8",
    "observedAt": "2026-09-16T12:40:26.331086+00:00"
  },
  "views_growth": {
    "question": "Quelle vidéo a gagné le plus de vues entre août et septembre ?",
    "title": "Comparer les gains de vues",
    "route": "sql_search",
    "path": [
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
      "end"
    ],
    "explanation": "La question demande une évolution du nombre de vues entre deux mois : la route analytique interroge l’historique des statistiques.",
    "steps": [
      "Le planner reconnaît une demande analytique.",
      "Le LLM Text-to-SQL produit une requête qui est validée puis exécutée.",
      "Les résultats structurés servent à générer la réponse.",
      "La conversation est mémorisée ; aucun embedding ni reranking n’intervient."
    ],
    "note": "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    "traceId": "6fffb70aba10d98d42889d9812154451",
    "observedAt": "2026-09-16T12:40:35.210342+00:00"
  },
  "gaelle": {
    "question": "J'ai besoin de la description de la vidéo de Gaëlle.",
    "title": "Retrouver la description de Gaëlle",
    "route": "sql_search",
    "path": [
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
      "end"
    ],
    "explanation": "La demande de description suit la route documentaire SQL, puis une restitution sans LLM de réponse.",
    "steps": [
      "Le planner identifie la personne et le document demandé.",
      "La résolution rapproche le nom des données connues, puis SQL retrouve les vidéos candidates.",
      "La description est récupérée et mise en forme sans LLM de réponse.",
      "Le document restitué et les sources rejoignent la conversation."
    ],
    "note": "Parcours observé dans Phoenix le 14/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    "traceId": "0f0555ade4537a426fd939682599e11b",
    "observedAt": "2026-09-14T21:22:18.920476+00:00"
  },
  "comparison": {
    "question": "Des points communs entre Sophie Ollivier et Chloé Leprètre ?",
    "title": "Comparer deux parcours",
    "route": "sql_search",
    "path": [
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
      "end"
    ],
    "explanation": "Les deux personnes identifiées dirigent la recherche vers SQL. La comparaison est ensuite rédigée à partir des vidéos retrouvées.",
    "steps": [
      "Le planner extrait les personnes mentionnées dans la question.",
      "La résolution rapproche les noms des intervenants et des mentions dans les transcripts.",
      "SQL retrouve les vidéos ciblées ; les résultats sont dédupliqués.",
      "Le LLM prépare la réponse à partir des sources, puis la conversation est mémorisée."
    ],
    "note": "Parcours observé dans Phoenix le 13/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    "traceId": "adfb6de489c57912b143668eeb7d0726",
    "observedAt": "2026-09-13T20:25:20.964642+00:00"
  },
  "ionis": {
    "question": "Pourquoi faire Ionis-STM ?",
    "title": "Explorer les raisons de choisir Ionis-STM",
    "route": "vector_search",
    "path": [
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
      "end"
    ],
    "explanation": "Cette question générale sur la formation emprunte la recherche hybride pour retrouver des passages pertinents.",
    "steps": [
      "Le reformulateur prépare la question, puis le planner définit la recherche.",
      "BM25 et pgvector retrouvent les passages ; RRF fusionne leurs classements.",
      "Le reranking Cohere est activé dans cette trace, puis le contexte hiérarchique est enrichi.",
      "Le LLM génère une réponse à partir des sources et la conversation est mémorisée."
    ],
    "note": "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    "traceId": "765c4241010d0a950f5ad46680affd1c",
    "observedAt": "2026-09-16T12:40:45.857312+00:00"
  },
  "ocean": {
    "question": "C'est quoi l'océan rouge ?",
    "title": "Expliquer l’océan rouge",
    "route": "vector_search",
    "path": [
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
      "end"
    ],
    "explanation": "La question porte sur une notion abordée dans les vidéos, sans personne ni document ciblé : la recherche est hybride.",
    "steps": [
      "Le reformulateur prépare la question, puis le planner définit la recherche.",
      "BM25 et pgvector retrouvent les passages ; RRF fusionne leurs classements.",
      "Le reranking Cohere est activé dans cette trace, puis le contexte hiérarchique est enrichi.",
      "Le LLM génère une réponse à partir des sources et la conversation est mémorisée."
    ],
    "note": "Parcours observé dans Phoenix le 13/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    "traceId": "7c3c473acacdfd6f07b65a3dbd36097a",
    "observedAt": "2026-09-13T19:27:52.943945+00:00"
  },
  "fadila": {
    "question": "Quelles questions ont été posées à Fadila ?",
    "title": "Retrouver les questions posées à Fadila",
    "route": "sql_search",
    "path": [
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
      "end"
    ],
    "explanation": "La personne identifiée permet de retrouver les vidéos correspondantes en SQL, puis d’en extraire les questions dans la réponse.",
    "steps": [
      "Le planner extrait les personnes mentionnées dans la question.",
      "La résolution rapproche les noms des intervenants et des mentions dans les transcripts.",
      "SQL retrouve les vidéos ciblées ; les résultats sont dédupliqués.",
      "Le LLM prépare la réponse à partir des sources, puis la conversation est mémorisée."
    ],
    "note": "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cet exemple enregistré ; elle ne relance pas le RAG.",
    "traceId": "2b8db3209a77e1aaeed15c224f42517d",
    "observedAt": "2026-09-16T16:54:15.371108+00:00"
  },
  "laura": {
    "question": "Qui est Laura Tyan ?",
    "title": "Utiliser la fiche de Laura Tyan",
    "route": "annex_direct",
    "path": [
      "start",
      "initialize",
      "reformulate",
      "plan",
      "annex_direct",
      "generate",
      "persist",
      "end"
    ],
    "explanation": "Laura Tyan figure dans les connaissances annexes. Dès que le planner identifie son nom, le pipeline utilise sa fiche locale.",
    "steps": [
      "La question est reformulée, puis le planner identifie Laura Tyan.",
      "Sa présence dans l’annexe déclenche la route annex_direct, avant la résolution des entités en base.",
      "Le LLM prépare la réponse à partir des connaissances locales ciblées, sans recherche SQL ni vectorielle.",
      "La réponse est finalisée et la conversation est mémorisée."
    ],
    "note": "Parcours observé dans Phoenix le 16/09/2026. La sélection affiche cette trace enregistrée ; elle ne relance pas le RAG.",
    "traceId": "d015c3afa2aeb3550edb5bb5d5fc7fe6",
    "observedAt": "2026-09-16T21:55:49.298987+00:00"
  }
};
// Sorties LLM enregistrées avec les scénarios : le visualiseur reste autonome
// même si les traces Phoenix sont purgées ou indisponibles.
const recordedLlmOutputs = {
  aymerich: { reformulate: { follow_up: false, reformulated_question: "Qui est Aymerich ?", topic: "Aymerich" }, plan: { route: "search", analytics: false, query_text: "Qui est Aymerich ?", query_text_bm25: "Aymerich identité", title_hints: [], persons: ["Aymerich"], companies: [], published_after: null, published_before: null } },
  salim: { reformulate: { follow_up: false, reformulated_question: "Pouvez-vous fournir la transcription de la vidéo de Salim ?", topic: "transcription de la vidéo de Salim" }, plan: { route: "search", analytics: false, query_text: "Quelle est la transcription de la vidéo de Salim ?", query_text_bm25: "transcription vidéo Salim", title_hints: [], persons: ["Salim"], companies: [], published_after: null, published_before: null } },
  views: { reformulate: { follow_up: false, reformulated_question: "Quelle vidéo a le plus de vues ?", topic: "vidéos" }, plan: { route: "search", analytics: true, query_text: "Quelle vidéo a le plus de vues ?", query_text_bm25: "vidéo plus vues", title_hints: [], persons: [], companies: [], published_after: null, published_before: null } },
  views_growth: { reformulate: { follow_up: false, reformulated_question: "Quelle vidéo a gagné le plus de vues entre août et septembre ?", topic: "vidéos et vues entre août et septembre" }, plan: { route: "search", analytics: true, query_text: "Quelle vidéo a gagné le plus de vues entre août et septembre ?", query_text_bm25: "vidéo plus vues août septembre", title_hints: [], persons: [], companies: [], published_after: null, published_before: null } },
  gaelle: { reformulate: { follow_up: false, reformulated_question: "Quelle est la description de la vidéo de Gaëlle ?", topic: "description de la vidéo de Gaëlle" }, plan: { route: "search", analytics: false, query_text: "Quelle est la description de la vidéo de Gaëlle ?", query_text_bm25: "description vidéo Gaëlle", title_hints: [], persons: ["Gaëlle"], companies: [], published_after: null, published_before: null } },
  comparison: { reformulate: { follow_up: false, reformulated_question: "Quels sont les points communs entre Sophie Ollivier et Chloé Leprètre ?", topic: "points communs entre Sophie Ollivier et Chloé Leprètre" }, plan: { route: "search", analytics: false, query_text: "Quels sont les points communs entre Sophie Ollivier et Chloé Leprètre ?", query_text_bm25: "points communs Sophie Ollivier Chloé Leprètre", title_hints: [], persons: ["Sophie Ollivier", "Chloé Leprètre"], companies: [], published_after: null, published_before: null } },
  ionis: { reformulate: { follow_up: false, reformulated_question: "Pourquoi choisir de faire des études à Ionis-STM ?", topic: "choix des études à Ionis-STM" }, plan: { route: "search", analytics: false, query_text: "Pourquoi choisir de faire des études à Ionis-STM ?", query_text_bm25: "études Ionis-STM raisons choix", title_hints: [], persons: [], companies: ["Ionis-STM"], published_after: null, published_before: null } },
  ocean: { reformulate: { follow_up: false, reformulated_question: "Qu'est-ce que l'océan rouge ?", topic: "océan rouge" }, plan: { route: "search", analytics: false, analytics_scope: null, analytics_metric: null, analytics_order: null, analytics_rank_start: null, analytics_rank_end: null, query_text: "Qu'est-ce que l'océan rouge ?", query_text_bm25: "océan rouge définition", title_hints: [], persons: [], companies: [], published_after: null, published_before: null } },
  fadila: { reformulate: { follow_up: false, reformulated_question: "Quelles questions ont été posées à Fadila ?", topic: "questions posées à Fadila" }, plan: { route: "search", analytics: false, query_text: "Quelles questions ont été posées à Fadila ?", query_text_bm25: "questions posées Fadila", title_hints: [], persons: ["Fadila"], companies: [], published_after: null, published_before: null } },
  laura: { reformulate: { follow_up: false, reformulated_question: "Qui est Laura Tyan ?", topic: "Laura Tyan" }, plan: { route: "search", analytics: false, query_text: "Qui est Laura Tyan ?", query_text_bm25: "Laura Tyan identité biographie", title_hints: [], persons: ["Laura Tyan"], companies: [], published_after: null, published_before: null } },
};
const staticStepOutputs = {
  initialize: "La requête, la conversation et les modèles sont préparés.",
  resolve_entities: "Les personnes, entreprises et titres extraits sont rapprochés des données connues.",
  build_execution_plan: "Le plan du LLM est complété par les règles de routage déterministes.",
  annex_direct: "La fiche locale correspondante est ajoutée au contexte de réponse.",
  person_clarification: "Une demande de précision est préparée pour identifier la personne visée.",
  direct: "La réponse sociale déjà prête est sélectionnée, sans recherche documentaire.",
  sql_search: "Les données structurées et les vidéos candidates sont recherchées en SQL.",
  vector_search: "La recherche hybride prépare les requêtes BM25 et vectorielle.",
  analytics: "La requête Text-to-SQL est validée puis exécutée avec des droits de lecture seuls.",
  lookup: "Les documents, descriptions ou transcriptions demandés sont récupérés.",
  retrieval: "Les passages sont classés, fusionnés puis enrichis de leur contexte hiérarchique.",
  generate: "Le LLM produit la réponse à partir du contexte et des sources retenues.",
  format: "La réponse ou le document déjà prêt est mis en forme sans appel LLM.",
  persist: "La réponse, les sources et le contexte de conversation sont enregistrés.",
  end: "La réponse finale est renvoyée à l’utilisateur.",
};
const escapeHtml = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const edgeId = (from, to) => `${from}--${to}`;
function edgePath(from, to) {
  const a = byId[from], b = byId[to];
  const sy = a.y + a.h, ey = b.y;
  // Route the long shortcuts outside the intermediate nodes.
  if (from === "plan" && to === "annex_direct") return `M ${a.x} ${sy} C ${a.x} ${sy+35}, 120 ${sy+35}, 120 ${ey}`;
  if (from === "vector_search") return `M 1040 ${sy} C 1130 ${sy+20}, 1130 ${ey-25}, 1040 ${ey}`;
  if (from === "retrieval") return `M 930 683 C 850 683, 530 660, 455 734`;
  if (from === "direct") return `M 580 ${sy} C 580 665, 700 655, 700 ${ey}`;
  if (from === "lookup" && to === "generate") return `M 905 ${sy} C 905 650, 430 650, 430 ${ey}`;
  return `M ${a.x} ${sy} C ${a.x} ${(sy+ey)/2}, ${b.x} ${(sy+ey)/2}, ${b.x} ${ey}`;
}
const svg = document.querySelector("#pipeline-graph");
svg.setAttribute("viewBox", "0 0 1160 925");
svg.innerHTML = `<title id="graph-title">Graphe complet du pipeline de réponse RAG</title><desc id="graph-description"></desc>
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="#383838" stroke-width="1.3"/></marker><marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="#ffffff" stroke-width="1.6"/></marker></defs>
  <g class="edges">${edges.map(([from,to]) => `<path id="${edgeId(from,to)}" class="edge" d="${edgePath(from,to)}" marker-end="url(#arrow)"/>`).join("")}</g>
  <g class="nodes">${nodes.map(n => `<g id="node-${n.id}" class="node ${n.terminal ? "terminal" : ""}" transform="translate(${n.x-n.w/2},${n.y})"><rect width="${n.w}" height="${n.h}" rx="7"/><text x="${n.w/2}" y="${n.sub ? 20 : 20}">${escapeHtml(n.label)}</text>${n.sub ? `<text class="subtitle" x="${n.w/2}" y="36">${escapeHtml(n.sub)}</text>` : ""}${n.extra ? `<text class="subtitle" x="${n.w/2}" y="52">${escapeHtml(n.extra)}</text>` : ""}</g>`).join("")}</g>`;
const select = document.querySelector("#scenario");
select.innerHTML = Object.entries(scenarios).map(([id, s]) => `<option value="${id}">${escapeHtml(s.question)}</option>`).join("");

function render() {
  const s = scenarios[select.value];
  const activeNodes = new Set(s.path);
  const activeEdges = new Set(s.path.slice(1).map((id, index) => edgeId(s.path[index], id)));
  nodes.forEach(n => document.getElementById(`node-${n.id}`).classList.toggle("active", activeNodes.has(n.id)));
  edges.forEach(([from,to]) => {
    const line = document.getElementById(edgeId(from,to));
    const active = activeEdges.has(line.id);
    line.classList.toggle("active", active);
    line.setAttribute("marker-end", `url(#${active ? "arrow-active" : "arrow"})`);
  });
  // Paint lit edges last so crossing inactive branches cannot obscure the route.
  svg.querySelectorAll(".edge.active").forEach(line => line.parentNode.append(line));
  document.querySelector("#graph-description").textContent = `Question : ${s.question}. Parcours éclairé : ${s.path.map(id => byId[id].label).join(" → ")}. Toutes les autres branches restent visibles en grisé.`;
  document.querySelector("#scenario-details").innerHTML = `<span class="eyebrow">Parcours de la question</span><h2>${escapeHtml(s.title)}</h2><span class="route-tag">${s.route}</span><p>${escapeHtml(s.explanation)}</p><h3>Ce qui se passe</h3><ol>${s.steps.map(step => `<li>${escapeHtml(step)}</li>`).join("")}</ol><div class="context"><h3>${s.traceId ? "Parcours observé" : "Règle vérifiée"}</h3><p>${escapeHtml(s.note)}</p></div>`;
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
    outputMarkup = "<p>Aucune sortie : cette étape n'est pas empruntée par la question sélectionnée.</p>";
  } else if (id === "reformulate" || id === "plan") {
    const output = recordedLlmOutputs[select.value]?.[id];
    outputMarkup = output
      ? `<pre>${escapeHtml(JSON.stringify(output, null, 2))}</pre>`
      : "<p>Sortie enregistrée indisponible pour cet exemple.</p>";
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
function scheduleHide() { closeTimer = setTimeout(hideTooltip, 140); }
svg.querySelectorAll(".node").forEach(node => {
  node.setAttribute("tabindex", "0");
  node.setAttribute("aria-label", byId[node.id.slice(5)].label);
  node.addEventListener("mouseenter", () => showTooltip(node));
  node.addEventListener("mouseleave", scheduleHide);
  node.addEventListener("focus", () => showTooltip(node));
  node.addEventListener("blur", hideTooltip);
});
tooltip.addEventListener("mouseenter", () => clearTimeout(closeTimer));
tooltip.addEventListener("mouseleave", scheduleHide);
document.addEventListener("keydown", event => { if (event.key === "Escape") hideTooltip(); });
select.addEventListener("change", hideTooltip);
window.addEventListener("resize", hideTooltip);
window.addEventListener("scroll", event => { if (!tooltip.contains(event.target)) hideTooltip(); }, true);
