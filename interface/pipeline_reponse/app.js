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
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="#455066" stroke-width="1.3"/></marker><marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="#6fe0ca" stroke-width="1.6"/></marker></defs>
  <g class="edges">${edges.map(([from,to]) => `<path id="${edgeId(from,to)}" class="edge" d="${edgePath(from,to)}" marker-end="url(#arrow)"/>`).join("")}</g>
  <g class="nodes">${nodes.map(n => `<g id="node-${n.id}" class="node ${n.terminal ? "terminal" : ""}" transform="translate(${n.x-n.w/2},${n.y})"><title>${escapeHtml(n.label)}${n.sub ? ` — ${escapeHtml(n.sub)}` : ""}</title><rect width="${n.w}" height="${n.h}" rx="7"/><text x="${n.w/2}" y="${n.sub ? 20 : 20}">${escapeHtml(n.label)}</text>${n.sub ? `<text class="subtitle" x="${n.w/2}" y="36">${escapeHtml(n.sub)}</text>` : ""}${n.extra ? `<text class="subtitle" x="${n.w/2}" y="52">${escapeHtml(n.extra)}</text>` : ""}</g>`).join("")}</g>`;
const select = document.querySelector("#scenario");
select.innerHTML = Object.entries(scenarios).map(([id, s]) => `<option value="${id}">${escapeHtml(s.question)}</option>`).join("");
const traceStates = new Map();
const outputSpanPrefixes = {
  start: ["request"],
  initialize: [],
  reformulate: ["reformulation"],
  plan: ["planner"],
  resolve_entities: ["person_resolution", "company_resolution", "title_resolution"],
  build_execution_plan: ["execution_plan"],
  annex_direct: ["orchestration"],
  person_clarification: ["orchestration"],
  direct: ["orchestration"],
  sql_search: ["sql"],
  vector_search: ["retrieval."],
  analytics: ["analytics."],
  lookup: ["person_in_", "company_in_", "title_lookup", "deduplicate_videos"],
  retrieval: ["retrieval."],
  generate: ["generation"],
  format: [],
  persist: ["store_message", "conversation_memory.update"],
  end: ["request"],
};

async function loadTrace(scenario) {
  if (!scenario.traceId || traceStates.has(scenario.traceId)) return;
  traceStates.set(scenario.traceId, { loading: true, spans: [] });
  try {
    const response = await fetch(`/api/pipeline-reponse/traces/${scenario.traceId}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const trace = await response.json();
    traceStates.set(scenario.traceId, { loading: false, spans: trace.spans || [], error: trace.error });
  } catch (error) {
    traceStates.set(scenario.traceId, { loading: false, spans: [], error: String(error) });
  }
  if (tooltipNode && scenarios[select.value].traceId === scenario.traceId) showTooltip(tooltipNode);
}

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
  loadTrace(s);
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
  const traceState = scenario.traceId ? traceStates.get(scenario.traceId) : null;
  let outputMarkup = "";
  if (!active) {
    outputMarkup = "<p>Aucune sortie : cette étape n'est pas empruntée par la question sélectionnée.</p>";
  } else if (!scenario.traceId) {
    outputMarkup = "<p>Aucune trace Phoenix n'est disponible pour ce scénario.</p>";
  } else if (!traceState || traceState.loading) {
    outputMarkup = "<p>Chargement de la sortie Phoenix…</p>";
  } else if (traceState.error) {
    outputMarkup = `<p>Impossible de charger la sortie Phoenix : ${escapeHtml(traceState.error)}</p>`;
  } else {
    const prefixes = outputSpanPrefixes[id] || [];
    const matchingSpans = traceState.spans.filter(span =>
      prefixes.some(prefix => span.name === prefix || span.name.startsWith(prefix))
    );
    if (!matchingSpans.length) {
      outputMarkup = "<p>Aucune sortie télémétrique n'a été enregistrée pour cette étape.</p>";
    } else {
      const output = matchingSpans.length === 1
        ? matchingSpans[0].output
        : Object.fromEntries(matchingSpans.map(span => [span.name, span.output]));
      outputMarkup = `<pre>${escapeHtml(JSON.stringify(output, null, 2))}</pre>`;
    }
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
  node.querySelector("title")?.remove();
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
