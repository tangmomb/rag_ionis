const $ = (selector) => document.querySelector(selector);
const state = { videos: [], selected: null, detail: null, tab: "summary" };

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char]));
}

async function getJson(url) {
  const response = await fetch(url);
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
  $("#videoBadges").innerHTML = [video.stage, video.video_type, video.has_subtitles === true ? "Sous-titres détectés" : video.has_subtitles === false ? "Sans sous-titres" : null]
    .filter(Boolean).map((value) => `<span class="badge">${escapeHtml(value)}</span>`).join("");
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
  $("#panel-summary").innerHTML = video.summary ? `<article class="prose">${renderMarkdown(video.summary)}</article>` : emptyPanel("Le résumé n’est pas encore produit.");
  $("#panel-transcript").innerHTML = video.transcript ? `<pre class="transcript">${escapeHtml(video.transcript)}</pre>` : emptyPanel("Le transcript n’est pas encore produit.");
  renderOcr(video.ocr);
  renderImages(video.images);
  renderChunks(video.chunks);
  renderFiles(video.files);
  switchTab(state.tab);
}

function ocrCount(groups) {
  return Object.values(groups || {}).reduce((total, group) => total + (Array.isArray(group) ? group.length : group && typeof group === "object" ? Object.keys(group).length : 0), 0);
}

function renderMarkdown(value) {
  const lines = value.split("\n");
  const output = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index].trim();
    if (line.startsWith("|") && /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+/.test(lines[index + 1] || "")) {
      const rows = [line];
      index += 2;
      while (index < lines.length && lines[index].trim().startsWith("|")) rows.push(lines[index++].trim());
      index -= 1;
      const cells = rows.map((row) => row.replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim()));
      output.push(`<div class="markdown-table-wrap"><table class="markdown-table"><thead><tr>${cells[0].map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>${cells.slice(1).map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
    } else if (line.startsWith("### ")) output.push(`<h3>${inlineMarkdown(line.slice(4))}</h3>`);
    else if (line.startsWith("## ")) output.push(`<h2>${inlineMarkdown(line.slice(3))}</h2>`);
    else if (line.startsWith("# ")) output.push(`<h2>${inlineMarkdown(line.slice(2))}</h2>`);
    else if (line.startsWith("- ")) output.push(`<p class="bullet">• ${inlineMarkdown(line.slice(2))}</p>`);
    else if (line) output.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  return output.join("");
}

function inlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
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
