const API_BASE = "https://www.googleapis.com/youtube/v3";

const requests = [
  {
    id: "channel-id",
    chapter: "1. Identifier la chaine",
    title: "Chaine depuis le handle",
    endpoint: "channels",
    purpose: "Obtenir l'ID YouTube interne de la chaine IONIS-STM.",
    notes: "Correspond au chapitre 1 du markdown.",
    fields: [
      { name: "part", label: "part", value: "id" },
      { name: "forHandle", label: "forHandle", value: "IONIS-STM" },
    ],
  },
  {
    id: "uploads-playlist",
    chapter: "2. Playlist uploads",
    title: "Playlist automatique des uploads",
    endpoint: "channels",
    purpose: "Obtenir l'ID de la playlist automatique qui contient toutes les videos publiees.",
    notes: "Utilise l'ID de chaine obtenu au chapitre 1.",
    fields: [
      { name: "part", label: "part", value: "contentDetails" },
      { name: "id", label: "channel_id", value: "" },
    ],
  },
  {
    id: "playlist-items",
    chapter: "3. Videos de la playlist",
    title: "Lister les IDs des videos",
    endpoint: "playlistItems",
    purpose: "Parcourir les videos de la playlist uploads, page par page.",
    notes: "maxResults est limite a 50 par l'API YouTube.",
    fields: [
      { name: "part", label: "part", value: "contentDetails" },
      { name: "playlistId", label: "uploads_playlist_id", value: "" },
      { name: "maxResults", label: "maxResults", value: "50" },
      { name: "pageToken", label: "pageToken", value: "" },
    ],
  },
  {
    id: "videos",
    chapter: "4. Metadonnees et stats",
    title: "Details des videos",
    endpoint: "videos",
    purpose: "Recuperer titre, description, duree, date de publication, vues, likes et commentaires.",
    notes: "Le champ id accepte une liste de 50 IDs video maximum, separes par des virgules.",
    fields: [
      { name: "part", label: "part", value: "snippet,contentDetails,statistics" },
      { name: "id", label: "video_ids", value: "", multiline: true },
      { name: "maxResults", label: "maxResults", value: "50" },
    ],
  },
  {
    id: "comment-threads",
    chapter: "5. Commentaires principaux",
    title: "Comment threads",
    endpoint: "commentThreads",
    purpose: "Recuperer les commentaires principaux d'une video et les premieres reponses incluses.",
    notes: "Certaines videos renvoient commentsDisabled si les commentaires sont fermes.",
    fields: [
      { name: "part", label: "part", value: "snippet,replies" },
      { name: "videoId", label: "youtube_video_id", value: "" },
      { name: "maxResults", label: "maxResults", value: "100" },
      { name: "pageToken", label: "pageToken", value: "" },
      { name: "textFormat", label: "textFormat", value: "plainText" },
    ],
  },
  {
    id: "comment-replies",
    chapter: "6. Reponses d'un commentaire",
    title: "Replies completes",
    endpoint: "comments",
    purpose: "Recuperer toutes les reponses d'un commentaire parent.",
    notes: "Utilise parentId, qui correspond a l'ID YouTube du commentaire principal.",
    fields: [
      { name: "part", label: "part", value: "snippet" },
      { name: "parentId", label: "parent_comment_id", value: "" },
      { name: "maxResults", label: "maxResults", value: "100" },
      { name: "pageToken", label: "pageToken", value: "" },
      { name: "textFormat", label: "textFormat", value: "plainText" },
    ],
  },
  {
    id: "transcripts",
    chapter: "7. Transcription",
    title: "Transcription video",
    endpoint: null,
    purpose: "Les transcriptions ne sont pas disponibles via YouTube Data API avec une simple API key.",
    notes:
      "Le script Python utilise youtube-transcript-api, une source non officielle. Cette console ne peut pas appeler cette librairie depuis le navigateur.",
    fields: [{ name: "video_id", label: "video_id", value: "" }],
    disabled: true,
  },
];

const requestList = document.querySelector("#requestList");
const chapterLabel = document.querySelector("#chapterLabel");
const requestTitle = document.querySelector("#requestTitle");
const requestPurpose = document.querySelector("#requestPurpose");
const methodBadge = document.querySelector("#methodBadge");
const fields = document.querySelector("#fields");
const urlPreview = document.querySelector("#urlPreview");
const notes = document.querySelector("#notes");
const responseOutput = document.querySelector("#responseOutput");
const statusBadge = document.querySelector("#statusBadge");
const sendButton = document.querySelector("#sendButton");
const apiKey = document.querySelector("#apiKey");
const toggleKey = document.querySelector("#toggleKey");
const copyUrl = document.querySelector("#copyUrl");

let selected = requests[0];

apiKey.addEventListener("input", () => {
  renderUrl();
});

toggleKey.addEventListener("click", () => {
  apiKey.type = apiKey.type === "password" ? "text" : "password";
});

copyUrl.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(buildUrl({ masked: false }));
    copyUrl.textContent = "Copie";
    setTimeout(() => {
      copyUrl.textContent = "Copier";
    }, 1200);
  } catch {
    setStatus("Copie indisponible", "error");
  }
});

sendButton.addEventListener("click", sendRequest);

function renderRequestList() {
  requestList.innerHTML = "";
  for (const request of requests) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `request-item${request.id === selected.id ? " active" : ""}`;
    button.innerHTML = `<span>${request.chapter}</span><strong>${request.title}</strong>`;
    button.addEventListener("click", () => {
      selected = request;
      render();
    });
    requestList.appendChild(button);
  }
}

function render() {
  renderRequestList();
  chapterLabel.textContent = selected.chapter;
  requestTitle.textContent = selected.title;
  requestPurpose.textContent = selected.purpose;
  methodBadge.textContent = selected.endpoint ? "GET" : "INFO";
  sendButton.disabled = Boolean(selected.disabled);
  statusBadge.className = "badge badge-muted";
  statusBadge.textContent = selected.disabled ? "Non appelable" : "En attente";
  responseOutput.textContent = selected.disabled
    ? "Cette etape ne correspond pas a une requete YouTube Data API officielle executable avec une API key."
    : "";
  notes.textContent = selected.notes;
  renderFields();
  renderUrl();
}

function renderFields() {
  fields.innerHTML = "";
  for (const field of selected.fields) {
    const label = document.createElement("label");
    label.className = "field";
    const caption = document.createElement("span");
    caption.textContent = field.label;
    const input = field.multiline ? document.createElement("textarea") : document.createElement("input");
    input.value = field.value;
    input.dataset.name = field.name;
    input.spellcheck = false;
    input.addEventListener("input", () => {
      field.value = input.value;
      renderUrl();
    });
    label.append(caption, input);
    fields.appendChild(label);
  }
}

function paramsFromFields({ masked }) {
  const params = new URLSearchParams();
  for (const field of selected.fields) {
    if (!field.value && field.name !== "pageToken") {
      continue;
    }
    params.set(field.name, field.value.replace(/\s+/g, field.multiline ? "" : " ").trim());
  }
  params.set("key", masked ? "***" : apiKey.value.trim());
  return params;
}

function buildUrl({ masked }) {
  if (!selected.endpoint) {
    return "Non disponible via YouTube Data API avec une simple API key.";
  }
  return `${API_BASE}/${selected.endpoint}?${paramsFromFields({ masked }).toString()}`;
}

function renderUrl() {
  urlPreview.textContent = buildUrl({ masked: true });
}

async function sendRequest() {
  if (!apiKey.value.trim()) {
    setStatus("Cle API manquante", "error");
    responseOutput.textContent = "Renseigne une cle API YouTube avant d'envoyer la requete.";
    return;
  }

  setStatus("Envoi", "muted");
  responseOutput.textContent = "";

  try {
    const response = await fetch(buildUrl({ masked: false }));
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.json() : await response.text();

    setStatus(`${response.status} ${response.statusText}`, response.ok ? "ok" : "error");
    responseOutput.textContent = typeof body === "string" ? body : JSON.stringify(body, null, 2);
  } catch (error) {
    setStatus("Erreur", "error");
    responseOutput.textContent = String(error);
  }
}

function setStatus(text, state) {
  statusBadge.textContent = text;
  statusBadge.className = `badge ${state === "ok" ? "badge-ok" : state === "error" ? "badge-error" : "badge-muted"}`;
}

render();
