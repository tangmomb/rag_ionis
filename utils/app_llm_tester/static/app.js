const providerColors = {
  openai: "#10a37f",
  mistral: "#f97316",
  google: "#4285f4",
};

const state = {
  providers: [],
  selectedProvider: "google",
  loading: false,
};

const elements = {
  form: document.querySelector("#request-form"),
  providerRow: document.querySelector("#provider-row"),
  model: document.querySelector("#model"),
  customModel: document.querySelector("#custom-model"),
  systemMessage: document.querySelector("#system-message"),
  message: document.querySelector("#message"),
  maxOutputTokens: document.querySelector("#max-output-tokens"),
  reasoningField: document.querySelector("#reasoning-field"),
  reasoningEffort: document.querySelector("#reasoning-effort"),
  reasoningHelp: document.querySelector("#reasoning-help"),
  verbosityField: document.querySelector("#verbosity-field"),
  verbosity: document.querySelector("#verbosity"),
  verbosityHelp: document.querySelector("#verbosity-help"),
  thinkingBudgetField: document.querySelector("#thinking-budget-field"),
  thinkingBudget: document.querySelector("#thinking-budget"),
  regionField: document.querySelector("#region-field"),
  region: document.querySelector("#region"),
  serviceTierField: document.querySelector("#service-tier-field"),
  serviceTier: document.querySelector("#service-tier"),
  characterCount: document.querySelector("#character-count"),
  formMessage: document.querySelector("#form-message"),
  submitButton: document.querySelector("#submit-button"),
  plainResponse: document.querySelector("#plain-response"),
  jsonResponse: document.querySelector("#json-response"),
  responseMeta: document.querySelector("#response-meta"),
  connectionState: document.querySelector("#connection-state"),
  toast: document.querySelector("#toast"),
};

function selectedProvider() {
  return state.providers.find((provider) => provider.id === state.selectedProvider);
}

function renderProviders() {
  elements.providerRow.replaceChildren(
    ...state.providers.map((provider) => {
      const button = document.createElement("button");
      const selected = provider.id === state.selectedProvider;
      button.type = "button";
      button.className = `provider-button${selected ? " selected" : ""}`;
      button.dataset.provider = provider.id;
      button.setAttribute("aria-pressed", String(selected));
      button.style.setProperty("--provider-color", providerColors[provider.id]);

      const providerName = document.createElement("span");
      providerName.className = "provider-name";
      const dot = document.createElement("span");
      dot.className = "provider-dot";
      const label = document.createElement("span");
      label.textContent = provider.label;
      providerName.append(dot, label);

      const keyStatus = document.createElement("span");
      keyStatus.className = `key-status${provider.configured ? " configured" : ""}`;
      keyStatus.textContent = provider.configured ? "clé prête" : "clé absente";
      button.append(providerName, keyStatus);
      button.addEventListener("click", () => selectProvider(provider.id));
      return button;
    }),
  );
}

function renderModels() {
  const provider = selectedProvider();
  if (!provider) return;
  const options = [
    ...provider.models.map((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      return option;
    }),
  ];
  const customOption = document.createElement("option");
  customOption.value = "__custom__";
  customOption.textContent = "Autre modèle…";
  options.push(customOption);
  elements.model.replaceChildren(...options);
  elements.model.value = provider.defaultModel;
  elements.customModel.value = "";
  elements.customModel.hidden = true;
  renderGenerationControls(provider, elements.model.value);
}

function renderGenerationControls(provider, model) {
  const controls = provider.generationControls;
  renderGenerationControl(
    elements.reasoningField,
    elements.reasoningEffort,
    elements.reasoningHelp,
    controls.reasoning,
    "Valeur par défaut du fournisseur",
  );
  elements.thinkingBudgetField.hidden = !model.startsWith(
    controls.thinkingBudgetModelPrefix || "__unsupported__",
  );
  elements.thinkingBudget.value = "";
  renderGenerationControl(
    elements.verbosityField,
    elements.verbosity,
    elements.verbosityHelp,
    controls.verbosity,
    "Valeur par défaut du fournisseur",
  );
  renderRegions(provider);
  renderServiceTiers(provider);
}

function renderRegions(provider) {
  const regions = provider.regions || [];
  elements.regionField.hidden = regions.length === 0;
  elements.region.replaceChildren(
    ...regions.map((region) => {
      const option = document.createElement("option");
      option.value = region.id;
      option.textContent = region.label;
      return option;
    }),
  );
}

function renderServiceTiers(provider) {
  const tiers = provider.generationControls.serviceTiers || [];
  elements.serviceTierField.hidden = tiers.length === 0;
  elements.serviceTier.replaceChildren(
    ...tiers.map((tier) => {
      const option = document.createElement("option");
      option.value = tier.id;
      option.textContent = tier.label;
      return option;
    }),
  );
}

function renderGenerationControl(field, select, help, levels, defaultLabel) {
  field.hidden = levels.length === 0;
  select.replaceChildren();
  if (!levels.length) return;
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = defaultLabel;
  select.append(defaultOption);
  levels.forEach((level) => {
    const option = document.createElement("option");
    option.value = level;
    option.textContent = level;
    select.append(option);
  });
  help.textContent = "Option envoyée uniquement si elle est choisie.";
}

function selectedModel() {
  return elements.model.value === "__custom__"
    ? elements.customModel.value.trim()
    : elements.model.value;
}

function toggleCustomModel() {
  const custom = elements.model.value === "__custom__";
  elements.customModel.hidden = !custom;
  elements.customModel.required = custom;
  if (custom) {
    elements.customModel.focus();
  }
}

function handleModelChange() {
  toggleCustomModel();
  const provider = selectedProvider();
  if (provider) renderGenerationControls(provider, elements.model.value);
}

function selectProvider(providerId) {
  state.selectedProvider = providerId;
  renderProviders();
  renderModels();
  const provider = selectedProvider();
  elements.formMessage.className = "";
  elements.formMessage.textContent = provider.configured
    ? `${provider.label} est prêt.`
    : `Ajoute ${provider.keyNames.join(" ou ")} dans .env.`;
}

function updateCharacterCount() {
  const count = elements.message.value.length;
  elements.characterCount.textContent = `${count.toLocaleString("fr-FR")} caractère${count > 1 ? "s" : ""}`;
}

function setLoading(loading) {
  state.loading = loading;
  elements.submitButton.disabled = loading;
  elements.submitButton.querySelector("span").textContent = loading
    ? "Requête en cours…"
    : "Envoyer la requête";
}

function showPayload(text, payload, meta) {
  elements.plainResponse.textContent = text || "Aucun texte n’a pu être extrait du payload.";
  elements.plainResponse.classList.toggle("empty", !text);
  elements.jsonResponse.textContent = JSON.stringify(payload, null, 2);
  elements.jsonResponse.classList.remove("empty");
  elements.responseMeta.textContent = meta;
}

function errorDetails(errorPayload, fallback) {
  if (errorPayload?.detail && typeof errorPayload.detail === "object") {
    return errorPayload.detail;
  }
  return { message: fallback };
}

async function submitRequest(event) {
  event.preventDefault();
  if (state.loading) return;

  const message = elements.message.value.trim();
  const model = selectedModel();
  if (!message || !model) {
    elements.formMessage.className = "error";
    elements.formMessage.textContent = "Le modèle et le message sont obligatoires.";
    return;
  }

  setLoading(true);
  elements.formMessage.className = "";
  elements.formMessage.textContent = "Connexion au fournisseur…";
  const startedAt = performance.now();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: state.selectedProvider,
        model,
        message,
        system_message: elements.systemMessage.value.trim() || null,
        max_output_tokens: Number(elements.maxOutputTokens.value),
        reasoning_effort: elements.reasoningEffort.value || null,
        verbosity: elements.verbosity.value || null,
        thinking_budget: elements.thinkingBudget.value === ""
          ? null
          : Number(elements.thinkingBudget.value),
        openai_region: state.selectedProvider === "openai" ? elements.region.value : "global",
        openai_service_tier: state.selectedProvider === "openai"
          ? elements.serviceTier.value
          : null,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const details = errorDetails(payload, `Erreur HTTP ${response.status}`);
      showPayload("", details.response ?? details, "La requête a échoué");
      throw new Error(details.message || `Erreur HTTP ${response.status}`);
    }

    showPayload(
      payload.text,
      payload.response,
      `${payload.provider} · ${payload.model} · ${(payload.durationMs / 1000).toFixed(2)} s`,
    );
    elements.formMessage.textContent = `Réponse reçue en ${(
      (performance.now() - startedAt) /
      1000
    ).toFixed(2)} s.`;
  } catch (error) {
    elements.formMessage.className = "error";
    elements.formMessage.textContent = error.message || "La requête a échoué.";
  } finally {
    setLoading(false);
  }
}

async function copyResult(targetId) {
  const text = document.querySelector(`#${targetId}`).textContent;
  await navigator.clipboard.writeText(text);
  elements.toast.classList.add("visible");
  window.setTimeout(() => elements.toast.classList.remove("visible"), 1400);
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error(`Erreur HTTP ${response.status}`);
    const payload = await response.json();
    state.providers = payload.providers;
    if (!state.providers.some((provider) => provider.id === state.selectedProvider)) {
      state.selectedProvider = state.providers[0]?.id;
    }
    renderProviders();
    renderModels();
    const configuredCount = state.providers.filter((provider) => provider.configured).length;
    elements.connectionState.classList.add("ready");
    elements.connectionState.lastChild.textContent = ` ${configuredCount}/${state.providers.length} clés prêtes`;
    selectProvider(state.selectedProvider);
  } catch (error) {
    elements.connectionState.textContent = "Configuration indisponible";
    elements.formMessage.className = "error";
    elements.formMessage.textContent = error.message;
  }
}

elements.form.addEventListener("submit", submitRequest);
elements.model.addEventListener("change", handleModelChange);
elements.customModel.addEventListener("input", () => {
  const provider = selectedProvider();
  if (provider && elements.model.value === "__custom__") {
    renderGenerationControls(provider, elements.customModel.value.trim());
  }
});
elements.message.addEventListener("input", updateCharacterCount);
elements.message.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") {
    elements.form.requestSubmit();
  }
});
document.querySelectorAll(".copy-button").forEach((button) => {
  button.addEventListener("click", () => copyResult(button.dataset.copyTarget));
});

updateCharacterCount();
loadConfig();
