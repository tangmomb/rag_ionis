const $ = (selector) => document.querySelector(selector);
const state = { tables: [], selected: null, offset: 0, total: 0, limit: 50, query: "", timer: null };

async function getJson(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "Erreur serveur");
  return body;
}

async function init() {
  try {
    const health = await getJson("/api/health");
    $("#healthDot").classList.toggle("online", health.ok);
    $("#healthText").textContent = health.ok ? `Connecté · ${health.database}` : "Base inaccessible";
    state.tables = await getJson("/api/tables");
    renderTables();
    if (state.selected) {
      const selectedStillExists = state.tables.some(
        (item) => item.schema === state.selected.schema && item.name === state.selected.name,
      );
      if (selectedStillExists) {
        $("#title").textContent = state.selected.name;
        $("#empty").classList.add("hidden");
        $("#content").classList.remove("hidden");
        await loadRows();
      } else {
        state.selected = null;
        $("#content").classList.add("hidden");
        $("#empty").classList.remove("hidden");
      }
    }
  } catch (error) { $("#healthText").textContent = error.message; }
}

function renderTables() {
  const filter = $("#tableFilter").value.toLowerCase();
  $("#tableList").innerHTML = state.tables.filter((item) => `${item.schema}.${item.name}`.toLowerCase().includes(filter)).map((item) =>
    `<button class="table-item ${state.selected && state.selected.schema === item.schema && state.selected.name === item.name ? "active" : ""}" data-schema="${escapeHtml(item.schema)}" data-name="${escapeHtml(item.name)}"><span>${escapeHtml(item.schema)}</span><strong>${escapeHtml(item.name)}</strong><em>${Number(item.estimated_rows).toLocaleString("fr-FR")}</em></button>`
  ).join("");
  document.querySelectorAll(".table-item").forEach((button) => button.addEventListener("click", () => selectTable(button.dataset.schema, button.dataset.name)));
}

async function selectTable(schema, name) {
  state.selected = { schema, name }; state.offset = 0; state.query = ""; $("#search").value = ""; $("#title").textContent = name; $("#empty").classList.add("hidden"); $("#content").classList.remove("hidden"); renderTables(); await loadRows();
}

async function loadRows() {
  if (!state.selected) return; $("#loading").classList.remove("hidden"); $("#noRows").classList.add("hidden");
  const params = new URLSearchParams({ limit: state.limit, offset: state.offset, q: state.query });
  try { const data = await getJson(`/api/tables/${encodeURIComponent(state.selected.schema)}/${encodeURIComponent(state.selected.name)}?${params}`); state.total = Number(data.total); renderData(data); }
  catch (error) { $("#tbody").innerHTML = `<tr><td class="error" colspan="99">${escapeHtml(error.message)}</td></tr>`; }
  finally { $("#loading").classList.add("hidden"); }
}

function renderData(data) {
  $("#thead").innerHTML = `<tr>${data.columns.map((column) => `<th title="${escapeHtml(column.type)}">${escapeHtml(column.name)}</th>`).join("")}</tr>`;
  $("#tbody").innerHTML = data.rows.map((row, rowIndex) => `<tr>${data.columns.map((column, columnIndex) => `<td data-row="${rowIndex}" data-column="${columnIndex}">${formatValue(row[column.name])}</td>`).join("")}</tr>`).join("");
  document.querySelectorAll("#tbody td").forEach((cell) => cell.addEventListener("click", () => {
    const row = data.rows[Number(cell.dataset.row)];
    const column = data.columns[Number(cell.dataset.column)];
    $("#detailColumn").textContent = `${column.name} · ${column.type}`;
    $("#detailValue").innerHTML = formatDetail(row[column.name]);
    $("#cellDetail").classList.remove("hidden");
    const sqlText = findSql(row[column.name]);
    $("#sqlValue").innerHTML = sqlText ? highlightSql(formatSql(sqlText)) : "";
    $("#sqlDetail").classList.toggle("hidden", !sqlText);
  }));
  $("#noRows").classList.toggle("hidden", data.rows.length > 0); $("#columnsMeta").textContent = `${data.columns.length} colonnes`; $("#meta").textContent = `${data.total.toLocaleString("fr-FR")} lignes`;
  const first = data.total ? state.offset + 1 : 0, last = Math.min(state.offset + state.limit, data.total); $("#pageInfo").textContent = `${first}–${last} sur ${data.total.toLocaleString("fr-FR")}`; $("#previous").disabled = state.offset === 0; $("#next").disabled = state.offset + state.limit >= data.total;
}

function formatValue(value) { if (value === null || value === undefined) return `<span class="null">NULL</span>`; if (typeof value === "object") value = JSON.stringify(value); return escapeHtml(String(value)); }
function fullValue(value) { if (value === null || value === undefined) return "NULL"; if (typeof value === "object") return JSON.stringify(value, null, 2); return String(value); }
function formatDetail(value) {
  value = parseJsonString(value);
  if (value === null || value === undefined) return '<span class="json-null">NULL</span>';
  if (typeof value !== "object") return escapeHtml(String(value));
  const json = JSON.stringify(value, null, 2); let output = ""; let cursor = 0;
  const tokenPattern = /("(?:\\.|[^"\\])*")(\s*:)?|\b(?:true|false|null)\b|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/g;
  for (const match of json.matchAll(tokenPattern)) {
    output += escapeHtml(json.slice(cursor, match.index));
    const token = match[0];
    if (match[2]) output += `<span class="json-key">${escapeHtml(token)}</span>${match[2]}`;
    else if (token === "true" || token === "false") output += `<span class="json-boolean">${token}</span>`;
    else if (token === "null") output += `<span class="json-null">${token}</span>`;
    else if (token.startsWith('"')) output += `<span class="json-string">${escapeHtml(token)}</span>`;
    else output += `<span class="json-number">${token}</span>`;
    cursor = match.index + token.length;
  }
  return output + escapeHtml(json.slice(cursor));
}
function parseJsonString(value) {
  if (typeof value !== "string") return value;
  const candidate = value.trim();
  if (!(candidate.startsWith("{") || candidate.startsWith("[") || candidate.startsWith('"'))) return value;
  try { return JSON.parse(candidate); } catch { return value; }
}
function findSql(value) {
  const parsed = parseJsonString(value);
  if (parsed !== value) return findSql(parsed);
  if (!value || typeof value !== "object") return "";
  if (!Array.isArray(value)) {
    const sqlKey = Object.keys(value).find((key) => key.toLowerCase() === "sql");
    if (sqlKey && typeof value[sqlKey] === "string" && value[sqlKey].trim()) return value[sqlKey].trim();
  }
  for (const child of Object.values(value)) {
    const nested = findSql(child);
    if (nested) return nested;
  }
  return "";
}
function formatSql(raw) {
  const major = /\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|OFFSET|UNION ALL|UNION|RETURNING|VALUES|SET)\b/gi;
  const clauses = new Set(["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "OFFSET", "UNION", "UNION ALL", "RETURNING", "VALUES", "SET"]);
  const sql = raw.trim().replace(/\s+/g, " ").replace(/\s*,\s*/g, ", ").replace(/\(\s+/g, "(").replace(/\s+\)/g, ")").replace(major, (match) => match.toUpperCase());
  const tokens = sql.split(" "); const lines = []; let current = "";
  const flush = () => { if (current.trim()) lines.push(current.trim()); current = ""; };
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i]; const upper = token.toUpperCase();
    if (clauses.has(upper) || ["JOIN", "ON", "AND", "OR"].includes(upper)) { flush(); current = token; if (["GROUP", "ORDER"].includes(upper) && tokens[i + 1]?.toUpperCase() === "BY") current += ` ${tokens[++i]}`; }
    else current += `${current ? " " : ""}${token}`;
  }
  flush(); return lines.join("\n");
}
function highlightSql(sql) {
  const pattern = /(--[^\n]*|\/\*[\s\S]*?\*\/|'(?:''|[^'])*'|"(?:""|[^"])*"|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][\w$]*(?=\s*\())/g;
  const keywords = /\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|FULL|INNER|OUTER|CROSS|ON|AND|OR|GROUP|BY|ORDER|HAVING|LIMIT|OFFSET|UNION|ALL|RETURNING|VALUES|SET|AS|DESC|ASC|NULLS|FIRST|LAST|IS|NOT|IN|LIKE|ILIKE|CASE|WHEN|THEN|ELSE|END|TRUE|FALSE|NULL)\b/gi;
  let output = ""; let cursor = 0;
  for (const match of sql.matchAll(pattern)) {
    output += highlightSqlPlain(sql.slice(cursor, match.index), keywords);
    const token = match[0]; const escaped = escapeHtml(token);
    if (token.startsWith("--") || token.startsWith("/*")) output += `<span class="sql-comment">${escaped}</span>`;
    else if (token.startsWith("'") || token.startsWith('"')) output += `<span class="sql-string">${escaped}</span>`;
    else if (/^\d/.test(token)) output += `<span class="sql-number">${escaped}</span>`;
    else output += `<span class="sql-function">${escaped}</span>`;
    cursor = match.index + token.length;
  }
  return output + highlightSqlPlain(sql.slice(cursor), keywords);
}
function highlightSqlPlain(value, keywords) {
  return escapeHtml(value).replace(keywords, (keyword) => `<span class="sql-keyword">${keyword.toUpperCase()}</span>`);
}
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char])); }

$("#tableFilter").addEventListener("input", renderTables); $("#refresh").addEventListener("click", init); $("#pageSize").addEventListener("change", (event) => { state.limit = Number(event.target.value); state.offset = 0; loadRows(); });
$("#search").addEventListener("input", (event) => { clearTimeout(state.timer); state.timer = setTimeout(() => { state.query = event.target.value; state.offset = 0; loadRows(); }, 300); });
$("#previous").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadRows(); }); $("#next").addEventListener("click", () => { state.offset += state.limit; loadRows(); });
$("#closeDetail").addEventListener("click", () => $("#cellDetail").classList.add("hidden"));
$("#copySql").addEventListener("click", async () => { try { await navigator.clipboard.writeText($("#sqlValue").textContent); $("#copySql").textContent = "Copié"; setTimeout(() => $("#copySql").textContent = "Copier", 1200); } catch { $("#copySql").textContent = "Indisponible"; } });
init();
