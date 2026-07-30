const $ = (selector) => document.querySelector(selector);
const state = {
  videos: [],
  selected: null,
  detail: null,
  tab: "transcript",
  transcriptType: null,
  transcriptEditing: false,
  transcriptMessage: "",
  speakerEditing: false,
  speakerMessage: "",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char]));
}

async function getJson(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "Erreur serveur");
  return body;
}

async function putJson(url, payload) {
  const response = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "Erreur serveur");
  return body;
}

function mediaUrl(path) {
  if (!state.detail || !path) return "";
  return `/api/videos/${encodeURIComponent(state.detail.run)}/${encodeURIComponent(state.detail.id)}/media?path=${encodeURIComponent(path)}`;
}

function formatDuration(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return "Durée inconnue";
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function renderVideos() {
  const query = $("#videoFilter").value.trim().toLowerCase();
  const matches = state.videos.filter((video) => `${video.title} ${video.id}`.toLowerCase().includes(query));
  $("#videoList").innerHTML = matches.map((video) => `
    <button class="table-item video-item ${state.selected && state.selected.run === video.run && state.selected.id === video.id ? "active" : ""}" data-run="${escapeHtml(video.run)}" data-id="${escapeHtml(video.id)}">
      <strong>${escapeHtml(video.title)}</strong><span>${escapeHtml(video.id)}</span><em>${escapeHtml(video.stage)}</em>
    </button>`).join("") || '<p class="side-empty">Aucune vidéo</p>';
  document.querySelectorAll(".video-item").forEach((button) => button.addEventListener("click", () => selectVideo(button.dataset.run, button.dataset.id)));
}

async function loadVideos(keepSelection = true) {
  try {
    state.videos = await getJson("/api/videos");
    $("#healthText").textContent = `${state.videos.length} vidéo${state.videos.length > 1 ? "s" : ""} indexée${state.videos.length > 1 ? "s" : ""}`;
    renderVideos();
    if (keepSelection && state.selected) await selectVideo(state.selected.run, state.selected.id);
    else if (!state.selected && state.videos.length) {
      const requestedId = new URLSearchParams(window.location.search).get("video");
      const requested = state.videos.find((video) => video.id === requestedId) || state.videos[0];
      await selectVideo(requested.run, requested.id);
    }
  } catch (error) {
    $("#healthDot").classList.remove("online");
    $("#healthText").textContent = error.message;
  }
}

async function selectVideo(run, id) {
  state.selected = { run, id };
  state.speakerEditing = false;
  state.speakerMessage = "";
  state.transcriptEditing = false;
  state.transcriptMessage = "";
  const url = new URL(window.location.href);
  url.searchParams.set("video", id);
  window.history.replaceState({}, "", url);
  renderVideos();
  $("#title").textContent = "Chargement…";
  try {
    state.detail = await getJson(`/api/videos/${encodeURIComponent(run)}/${encodeURIComponent(id)}`);
    renderDetail();
  } catch (error) {
    $("#title").textContent = "Erreur";
    $("#empty").innerHTML = `<h2>Impossible de charger la vidéo</h2><p>${escapeHtml(error.message)}</p>`;
  }
}

function renderDetail() {
  const video = state.detail;
  $("#title").textContent = video.title;
  $("#empty").classList.add("hidden");
  $("#content").classList.remove("hidden");
  const standardBadges = [
    video.stage,
    video.video_type,
    video.has_subtitles === true ? "Sous-titres détectés" : video.has_subtitles === false ? "Sans sous-titres" : null,
  ].filter(Boolean).map((value) => `<span class="badge">${escapeHtml(value)}</span>`).join("");
  const speakers = Array.isArray(video.speakers) ? video.speakers : [];
  const speakerDetails = Array.isArray(video.speaker_details) && video.speaker_details.length
    ? video.speaker_details
    : speakers.map((name) => ({ name, title: null }));
  const speakerBadges = speakerDetails.map((speaker) => {
    const label = speaker.title ? `${speaker.name} — ${speaker.title}` : speaker.name;
    return `<span class="badge speaker-badge" title="${escapeHtml(label)}"><span class="badge-info">i</span>${escapeHtml(label)}</span>`;
  }).join("");
  $("#videoBadges").innerHTML = standardBadges + speakerBadges;
  $("#speakerStatus").textContent = state.speakerMessage;
  renderSpeakerEditor(speakerDetails);
  $("#editSpeakers").onclick = () => {
    state.speakerEditing = true;
    state.speakerMessage = "";
    $("#speakerStatus").textContent = "";
    renderSpeakerEditor(speakerDetails);
  };
  $("#videoMeta").textContent = `${video.id} · ${formatDuration(video.duration_seconds)} · ${video.run}`;
  $("#youtubeLink").href = video.url;
  $("#stats").innerHTML = [
    [video.image_count, "images"], [ocrCount(video.ocr), "textes OCR"], [video.chunk_count, "chunks"], [video.embedding_count, "embeddings"],
  ].map(([value, label]) => `<div><strong>${Number(value).toLocaleString("fr-FR")}</strong><span>${label}</span></div>`).join("");
  if (video.video_path) {
    $("#preview").innerHTML = `<video controls preload="metadata" src="${mediaUrl(video.video_path)}"></video>`;
  } else if (video.preview_path) {
    $("#preview").innerHTML = `<img src="${mediaUrl(video.preview_path)}" alt="Aperçu de ${escapeHtml(video.title)}">`;
  } else {
    $("#preview").innerHTML = '<div class="preview-empty">Aucun aperçu</div>';
  }
  renderTranscripts(video.transcripts || []);
  renderOcr(video.ocr);
  renderImages(video.images);
  renderChunks(video.chunks);
  renderFiles(video.files);
  switchTab(state.tab);
}

function speakerFormValues() {
  return Array.from(document.querySelectorAll(".speaker-edit-row")).map((row) => ({
    name: row.querySelector("[data-field='name']").value,
    title: row.querySelector("[data-field='title']").value,
  }));
}

function renderSpeakerEditor(speakers) {
  const editor = $("#speakerEditor");
  if (!state.speakerEditing) {
    editor.classList.add("hidden");
    editor.innerHTML = "";
    return;
  }
  editor.classList.remove("hidden");
  editor.innerHTML = `
    <div class="speaker-editor-heading">
      <strong>Speakers validés</strong>
      <span>Nom et fonction</span>
    </div>
    <div class="speaker-edit-list">
      ${speakers.map((speaker, index) => `
        <div class="speaker-edit-row">
          <input data-field="name" aria-label="Nom du speaker ${index + 1}" maxlength="200" value="${escapeHtml(speaker.name)}" placeholder="Nom">
          <input data-field="title" aria-label="Fonction du speaker ${index + 1}" maxlength="300" value="${escapeHtml(speaker.title || "")}" placeholder="Fonction">
          <button class="icon-button remove-speaker" data-index="${index}" aria-label="Supprimer ${escapeHtml(speaker.name || `le speaker ${index + 1}`)}">×</button>
        </div>`).join("")}
    </div>
    <div class="edit-actions">
      <button id="addSpeaker" class="button secondary compact-button">+ Ajouter</button>
      <span class="actions-spacer"></span>
      <button id="cancelSpeakers" class="button secondary compact-button">Annuler</button>
      <button id="saveSpeakers" class="button compact-button" data-testid="save-speakers">Enregistrer</button>
    </div>
    <p id="speakerEditError" class="edit-error" role="alert"></p>`;

  document.querySelectorAll(".remove-speaker").forEach((button) => button.addEventListener("click", () => {
    const values = speakerFormValues();
    values.splice(Number(button.dataset.index), 1);
    renderSpeakerEditor(values);
  }));
  $("#addSpeaker").addEventListener("click", () => {
    renderSpeakerEditor([...speakerFormValues(), { name: "", title: "" }]);
  });
  $("#cancelSpeakers").addEventListener("click", () => {
    state.speakerEditing = false;
    renderSpeakerEditor(speakers);
  });
  $("#saveSpeakers").addEventListener("click", saveSpeakers);
}

async function saveSpeakers() {
  const button = $("#saveSpeakers");
  const error = $("#speakerEditError");
  button.disabled = true;
  error.textContent = "";
  try {
    const result = await putJson(
      `/api/videos/${encodeURIComponent(state.detail.run)}/${encodeURIComponent(state.detail.id)}/speakers`,
      { speakers: speakerFormValues() },
    );
    state.detail.speakers = result.speakers;
    state.detail.speaker_details = result.speaker_details;
    const overview = state.videos.find((video) => video.run === state.detail.run && video.id === state.detail.id);
    if (overview) {
      overview.speakers = result.speakers;
      overview.speaker_details = result.speaker_details;
    }
    state.speakerEditing = false;
    state.speakerMessage = `Enregistré dans ${result.path}`;
    renderDetail();
  } catch (saveError) {
    error.textContent = saveError.message;
    button.disabled = false;
  }
}

function renderTranscripts(transcripts) {
  const available = transcripts.filter((transcript) => transcript && transcript.content);
  if (!available.length) {
    $("#panel-transcript").innerHTML = state.detail.transcript
      ? `<pre class="transcript">${escapeHtml(state.detail.transcript)}</pre>`
      : emptyPanel("Le transcript n’est pas encore produit.");
    return;
  }
  if (!available.some((transcript) => transcript.key === state.transcriptType)) {
    state.transcriptType = available[0].key;
  }
  const selected = available.find((transcript) => transcript.key === state.transcriptType) || available[0];
  const canEdit = selected.key === "enriched" && selected.editable === true;
  const transcriptBody = state.transcriptEditing && canEdit
    ? `<textarea id="transcriptEditor" class="transcript-editor" data-testid="enriched-transcript-editor">${escapeHtml(selected.content)}</textarea>
       <div class="edit-actions transcript-edit-actions">
         <button id="cancelTranscript" class="button secondary compact-button">Annuler</button>
         <button id="saveTranscript" class="button compact-button" data-testid="save-enriched-transcript">Enregistrer</button>
       </div>
       <p id="transcriptEditError" class="edit-error" role="alert"></p>`
    : `<pre class="transcript">${escapeHtml(selected.content)}</pre>`;
  $("#panel-transcript").innerHTML = `
    <div class="transcript-types">
      ${available.map((transcript) => `
        <button class="transcript-type ${transcript.key === selected.key ? "active" : ""}" data-transcript-type="${escapeHtml(transcript.key)}">
          ${escapeHtml(transcript.label)}
        </button>`).join("")}
    </div>
    <div class="transcript-heading">
      <div>
        <strong>${escapeHtml(selected.label)}</strong>
        ${canEdit && !state.transcriptEditing ? '<button id="editTranscript" class="button secondary compact-button" data-testid="edit-enriched-transcript">Modifier</button>' : ""}
      </div>
      <span>${escapeHtml(selected.path)}</span>
    </div>
    ${state.transcriptMessage && selected.key === "enriched" ? `<p class="save-status transcript-save-status">${escapeHtml(state.transcriptMessage)}</p>` : ""}
    ${transcriptBody}`;
  document.querySelectorAll(".transcript-type").forEach((button) => button.addEventListener("click", () => {
    state.transcriptType = button.dataset.transcriptType;
    state.transcriptEditing = false;
    state.transcriptMessage = "";
    renderTranscripts(state.detail.transcripts || []);
  }));
  if (canEdit && !state.transcriptEditing) {
    $("#editTranscript").addEventListener("click", () => {
      state.transcriptEditing = true;
      state.transcriptMessage = "";
      renderTranscripts(state.detail.transcripts || []);
    });
  }
  if (state.transcriptEditing && canEdit) {
    $("#cancelTranscript").addEventListener("click", () => {
      state.transcriptEditing = false;
      renderTranscripts(state.detail.transcripts || []);
    });
    $("#saveTranscript").addEventListener("click", () => saveEnrichedTranscript(selected));
  }
}

async function saveEnrichedTranscript(selected) {
  const button = $("#saveTranscript");
  const error = $("#transcriptEditError");
  button.disabled = true;
  error.textContent = "";
  try {
    const result = await putJson(
      `/api/videos/${encodeURIComponent(state.detail.run)}/${encodeURIComponent(state.detail.id)}/transcripts/enriched`,
      { content: $("#transcriptEditor").value },
    );
    selected.content = result.content;
    selected.path = result.path;
    selected.editable = result.editable;
    const plainTranscript = (state.detail.transcripts || []).find((transcript) => transcript.key === "plain");
    if (plainTranscript && result.plain) {
      plainTranscript.path = result.plain.path;
      plainTranscript.content = result.plain.content;
    }
    if (result.plain) state.detail.transcript = result.plain.content;
    if (result.chunks) {
      state.detail.chunks = result.chunks.items;
      state.detail.chunk_count = result.chunks.items.length;
      const overview = state.videos.find((video) => video.run === state.detail.run && video.id === state.detail.id);
      if (overview) overview.chunk_count = result.chunks.items.length;
      renderChunks(result.chunks.items);
    }
    state.transcriptEditing = false;
    state.transcriptMessage = `Enregistré · transcript plain et ${result.chunks.items.length} chunk(s) régénérés`;
    renderTranscripts(state.detail.transcripts || []);
  } catch (saveError) {
    error.textContent = saveError.message;
    button.disabled = false;
  }
}

function ocrCount(groups) {
  return Object.values(groups || {}).reduce((total, group) => total + (Array.isArray(group) ? group.length : group && typeof group === "object" ? Object.keys(group).length : 0), 0);
}

function renderOcr(groups) {
  const sections = Object.entries(groups || {}).map(([kind, values]) => {
    const entries = Array.isArray(values) ? values.map((value, index) => [index + 1, typeof value === "object" ? value.text || value.content || JSON.stringify(value) : value]) : Object.entries(values || {});
    return `<section class="ocr-group"><h2>${escapeHtml(kind)} <span>${entries.length}</span></h2>${entries.map(([time, text]) => `<div class="ocr-line"><button data-time="${escapeHtml(String(time).split("#")[0])}" class="timecode">${escapeHtml(time)}</button><p>${escapeHtml(text)}</p></div>`).join("")}</section>`;
  }).join("");
  $("#panel-ocr").innerHTML = sections || emptyPanel("Les données OCR ne sont pas encore produites.");
  document.querySelectorAll(".timecode").forEach((button) => button.addEventListener("click", () => seekVideo(button.dataset.time)));
}

function seekVideo(timecode) {
  const parts = timecode.split(":").map(Number);
  if (parts.some((value) => !Number.isFinite(value))) return;
  const seconds = parts.reduce((total, value) => total * 60 + value, 0);
  const player = $("#preview video");
  if (player) { player.currentTime = seconds; player.play(); }
}

function renderImages(images) {
  $("#panel-images").innerHTML = images.length ? `<div id="selectedImage" class="selected-image hidden"></div><div class="image-grid">${images.map((image) => `<button class="image-tile" data-path="${escapeHtml(image.path)}"><img loading="lazy" src="${mediaUrl(image.path)}" alt="${escapeHtml(image.name)}"><span>${escapeHtml(image.category)} · ${escapeHtml(image.name)}</span></button>`).join("")}</div>` : emptyPanel("Les images ne sont pas encore produites.");
  document.querySelectorAll(".image-tile").forEach((button) => button.addEventListener("click", () => {
    const target = $("#selectedImage");
    target.innerHTML = `<img src="${mediaUrl(button.dataset.path)}" alt="Image sélectionnée"><p>${escapeHtml(button.dataset.path)}</p>`;
    target.classList.remove("hidden");
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  }));
}

function renderChunks(chunks) {
  $("#panel-chunks").innerHTML = chunks.length ? `<div class="chunk-list">${chunks.map((chunk, index) => {
    const speakers = chunk.meta_data && Array.isArray(chunk.meta_data.speakers) ? chunk.meta_data.speakers.join(", ") : "";
    return `<article class="chunk"><header><strong>Chunk ${escapeHtml(chunk.chunk_index || index + 1)}</strong><span>${escapeHtml(speakers)}</span><em>${Number(chunk.char_count || String(chunk.content || chunk.text || "").length).toLocaleString("fr-FR")} caractères</em></header><p>${escapeHtml(chunk.content || chunk.text || "")}</p></article>`;
  }).join("")}</div>` : emptyPanel("Les chunks ne sont pas encore produits.");
}

function renderFiles(files) {
  $("#panel-files").innerHTML = files.length ? `<div class="file-list">${files.map((file) => `<a href="${mediaUrl(file.path)}" target="_blank"><span>${escapeHtml(file.path)}</span><em>${formatBytes(file.size)}</em></a>`).join("")}</div>` : emptyPanel("Aucun fichier produit.");
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} o`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`;
  return `${(bytes / 1024 / 1024).toFixed(1)} Mo`;
}

function emptyPanel(message) { return `<div class="panel-empty">${escapeHtml(message)}</div>`; }

function switchTab(name) {
  state.tab = name;
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  document.querySelectorAll(".data-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== `panel-${name}`));
}

$("#videoFilter").addEventListener("input", renderVideos);
$("#refresh").addEventListener("click", () => loadVideos(true));
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => switchTab(tab.dataset.tab)));
loadVideos(false);
