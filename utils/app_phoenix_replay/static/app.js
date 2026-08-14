const form = document.querySelector("#replay-form");
const traceInput = document.querySelector("#trace-id");
const replayButton = document.querySelector("#replay-button");
const resultPanel = document.querySelector(".result-panel");

const states = {
  empty: document.querySelector("#empty-state"),
  loading: document.querySelector("#loading-state"),
  error: document.querySelector("#error-state"),
  inspection: document.querySelector("#inspection-state"),
  success: document.querySelector("#success-state"),
};

let inspectedTraceId = null;

function showState(name) {
  Object.entries(states).forEach(([key, element]) => {
    element.hidden = key !== name;
  });
  resultPanel.setAttribute("aria-busy", name === "loading" ? "true" : "false");
}

function showLoading(title, copy) {
  document.querySelector("#loading-title").textContent = title;
  document.querySelector("#loading-copy").textContent = copy;
  showState("loading");
}

function showError(message) {
  document.querySelector("#error-message").textContent = message;
  showState("error");
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : "Erreur inconnue.");
  }
  return data;
}

function renderInspection(data) {
  document.querySelector("#target-question").textContent = data.target_question;
  document.querySelector("#history-count").textContent = `${data.history_message_count} échange${data.history_message_count > 1 ? "s" : ""}`;
  document.querySelector("#original-conversation").textContent = data.original_conversation_id;
  document.querySelector("#target-message").textContent = data.target_message_id;

  const historyList = document.querySelector("#history-list");
  historyList.replaceChildren();
  data.history.forEach((message) => {
    const item = document.createElement("li");
    const question = document.createElement("strong");
    const answer = document.createElement("p");
    question.textContent = message.user_message;
    answer.textContent = message.answer_preview || "Réponse vide";
    item.append(question, answer);
    historyList.appendChild(item);
  });

  inspectedTraceId = data.original_trace_id;
  replayButton.disabled = false;
  showState("inspection");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  replayButton.disabled = true;
  inspectedTraceId = null;
  showLoading("Inspection du contexte…", "Lecture de la trace et de la conversation associée.");
  try {
    const data = await postJson("/api/inspect", {
      trace_id: traceInput.value.trim(),
      mode: "exact-context",
    });
    renderInspection(data);
  } catch (error) {
    showError(error.message);
  }
});

traceInput.addEventListener("input", () => {
  if (traceInput.value.trim().toLowerCase() !== inspectedTraceId) {
    replayButton.disabled = true;
  }
});

replayButton.addEventListener("click", async () => {
  if (!inspectedTraceId) return;
  const confirmed = window.confirm(
    "Rejouer ce tour ? Une nouvelle conversation sera créée et Mistral sera appelé."
  );
  if (!confirmed) return;

  replayButton.disabled = true;
  showLoading("Rejeu en cours…", "Le pipeline complet traite le dernier message avec le contexte cloné.");
  try {
    const data = await postJson("/api/replay", {
      trace_id: inspectedTraceId,
      mode: "exact-context",
    });
    document.querySelector("#replay-conversation").textContent = data.replay_conversation_id;
    document.querySelector("#replay-trace").textContent = data.replay_trace_id || "Télémétrie indisponible";
    document.querySelector("#replay-answer").textContent = data.response?.answer || "Réponse vide";
    document.querySelector("#open-phoenix").href = data.phoenix_url || "http://127.0.0.1:6006/projects";
    showState("success");
  } catch (error) {
    replayButton.disabled = false;
    showError(error.message);
  }
});
