const example = `SELECT c.id, v.title, v.url, c.chunk_index, c.content, ARRAY(SELECT s.name FROM speakers s WHERE s.video_id = v.id ORDER BY s.id) AS speakers, ts_rank_cd( to_tsvector('french', coalesce(c.content, '')), websearch_to_tsquery('french', %s) ) AS score FROM chunks c JOIN videos v ON v.id = c.video_id WHERE to_tsvector('french', coalesce(c.content, '')) @@ websearch_to_tsquery('french', %s) ORDER BY score DESC, c.id ASC LIMIT %s`;
const input = document.querySelector('#input');
const output = document.querySelector('#output code');
const status = document.querySelector('#status');
const inputCount = document.querySelector('#inputCount');
const keywords = /\b(SELECT|FROM|WHERE|JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN|INNER JOIN|OUTER JOIN|CROSS JOIN|ON|AND|OR|GROUP BY|ORDER BY|HAVING|LIMIT|OFFSET|UNION ALL|UNION|RETURNING|VALUES|SET|AS|DESC|ASC|NULLS FIRST|NULLS LAST)\b/gi;
const major = new Set(['SELECT','FROM','WHERE','GROUP BY','ORDER BY','HAVING','LIMIT','OFFSET','UNION','UNION ALL','RETURNING','VALUES','SET']);

function protectStrings(sql) {
  const strings = [];
  const protectedSql = sql.replace(/'(?:''|[^'])*'|"(?:""|[^"])*"|--[^\n]*|\/\*[\s\S]*?\*\//g, match => {
    strings.push(match); return `\u0000${strings.length - 1}\u0000`;
  });
  return [protectedSql, strings];
}
function restore(sql, strings) { return sql.replace(/\u0000(\d+)\u0000/g, (_, i) => strings[i]); }
function formatSql(raw, size) {
  let sql = raw.trim().replace(/\s+/g, ' ');
  if (!sql) return '';
  const [safe, strings] = protectStrings(sql);
  let text = safe.replace(/\s*,\s*/g, ', ').replace(/\(\s+/g, '(').replace(/\s+\)/g, ')');
  text = text.replace(keywords, match => match.toUpperCase());
  const indent = ' '.repeat(size), lines = [];
  const tokens = text.split(' '); let current = '', depth = 0;
  const flush = () => { if (current.trim()) lines.push(indent.repeat(Math.max(depth, 0)) + current.trim()); current = ''; };
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i], upper = token.toUpperCase();
    if (token.includes('(')) depth += (token.match(/\(/g) || []).length;
    if (token.includes(')')) depth -= (token.match(/\)/g) || []).length;
    if (major.has(upper) || (upper === 'LEFT' || upper === 'RIGHT' || upper === 'INNER' || upper === 'FULL' || upper === 'CROSS') && tokens[i + 1]?.toUpperCase() === 'JOIN') {
      if (upper !== 'ALL' && tokens[i - 1]?.toUpperCase() === 'UNION') { current += ` ${token}`; continue; }
      flush(); current = token;
      if ((upper === 'GROUP' || upper === 'ORDER') && tokens[i + 1]?.toUpperCase() === 'BY') { current += ` ${tokens[++i]}`; }
    } else if (upper === 'JOIN' || upper === 'ON' || upper === 'AND' || upper === 'OR') {
      flush(); current = token;
    } else { current += (current ? ' ' : '') + token; }
  }
  flush();
  return restore(lines.join('\n'), strings).replace(/\s+;/g, ';');
}
function escapeHtml(value) {
  return value.replace(/[&<>"']/g, character => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[character]));
}
function highlightSql(sql) {
  const tokenPattern = /(--[^\n]*|\/\*[\s\S]*?\*\/|'(?:''|[^'])*'|"(?:""|[^"])*"|%s|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][\w$]*(?=\s*\())/g;
  let highlighted = '', cursor = 0, match;
  while ((match = tokenPattern.exec(sql)) !== null) {
    highlighted += highlightPlain(sql.slice(cursor, match.index));
    const token = match[0], escaped = escapeHtml(token);
    if (token.startsWith('--') || token.startsWith('/*')) highlighted += `<span class="sql-comment">${escaped}</span>`;
    else if (token === '%s') highlighted += `<span class="sql-parameter">${escaped}</span>`;
    else if (token.startsWith("'") || token.startsWith('"')) highlighted += `<span class="sql-string">${escaped}</span>`;
    else if (/^\d/.test(token)) highlighted += `<span class="sql-number">${escaped}</span>`;
    else highlighted += `<span class="sql-function">${escaped}</span>`;
    cursor = match.index + token.length;
  }
  return highlighted + highlightPlain(sql.slice(cursor));
}
function highlightPlain(value) {
  const escaped = escapeHtml(value);
  return escaped.replace(/\b(SELECT|FROM|WHERE|JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN|INNER JOIN|OUTER JOIN|CROSS JOIN|ON|AND|OR|GROUP BY|ORDER BY|HAVING|LIMIT|OFFSET|UNION ALL|UNION|RETURNING|VALUES|SET|AS|DESC|ASC|NULLS FIRST|NULLS LAST|IS NULL|IS NOT NULL|NOT|IN|LIKE|ILIKE|CASE|WHEN|THEN|ELSE|END)\b/gi, keyword => `<span class="sql-keyword">${keyword.toUpperCase()}</span>`);
}
function render() { const result = formatSql(input.value, Number(document.querySelector('#indent').value)); output.innerHTML = result ? highlightSql(result) : 'Le résultat apparaîtra ici.'; status.textContent = result ? 'Requête formatée' : 'Prêt'; }
input.addEventListener('input', () => { inputCount.textContent = `${input.value.length.toLocaleString('fr-FR')} caractères`; });
document.querySelector('#format').addEventListener('click', render);
document.querySelector('#sample').addEventListener('click', () => { input.value = example; input.dispatchEvent(new Event('input')); render(); });
document.querySelector('#clear').addEventListener('click', () => { input.value = ''; input.dispatchEvent(new Event('input')); render(); input.focus(); });
document.querySelector('#indent').addEventListener('change', render);
document.querySelector('#copy').addEventListener('click', async () => { const text = output.textContent; if (!text || text.startsWith('Le résultat')) return; await navigator.clipboard.writeText(text); status.textContent = 'Copié dans le presse-papiers'; });
input.addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') render(); });
document.querySelector('#sample').click();
