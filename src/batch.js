export const BATCH_HISTORY_KEY = "refloor-lab:batch-history";
export const ABLATION_VERSION = "ablation-lab.v1";

const HISTORY_LIMIT = 250;

function bytesToHex(buffer) {
  return [...new Uint8Array(buffer)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function sha256Text(text) {
  const bytes = new TextEncoder().encode(text);
  return bytesToHex(await crypto.subtle.digest("SHA-256", bytes));
}

export async function createBatchJobs(files) {
  const createdAt = new Date().toISOString();
  const jobs = [];
  for (const file of files) {
    const text = await file.text();
    const hash = await sha256Text(text);
    jobs.push({
      id: `${hash.slice(0, 12)}-${createdAt}-${Math.random().toString(36).slice(2, 8)}`,
      name: file.name,
      size: file.size,
      hash,
      text,
      createdAt,
      status: "queued",
      progressPct: 0,
      progressStage: "Queued",
      metrics: {},
      removalCounts: { local: 0, scaffold: 0 },
      downloadStatus: "pending",
    });
  }
  return jobs;
}

export function loadBatchHistory() {
  try {
    const rows = JSON.parse(localStorage.getItem(BATCH_HISTORY_KEY) ?? "[]");
    return Array.isArray(rows) ? rows : [];
  } catch {
    return [];
  }
}

export function saveBatchHistory(rows) {
  const metadataOnly = rows.map(toHistoryRow).slice(0, HISTORY_LIMIT);
  localStorage.setItem(BATCH_HISTORY_KEY, JSON.stringify(metadataOnly));
}

export function clearBatchHistory() {
  localStorage.removeItem(BATCH_HISTORY_KEY);
}

export function toHistoryRow(row) {
  const {
    text,
    refloored,
    result,
    ...history
  } = row;
  return history;
}

export function countRemovalMethods(spans = []) {
  return spans.reduce((counts, span) => {
    const method = span.method === "scaffold" ? "scaffold" : "local";
    counts[method] += 1;
    return counts;
  }, { local: 0, scaffold: 0 });
}

function scaffoldModeName(outputMode) {
  if (outputMode === "pattern-ceiling") return "light";
  if (outputMode === "pattern-floor2") return "medium";
  if (outputMode === "pattern-floor1") return "hard";
  return "off";
}

export function prefixAblationFrontmatter(text, { fileName, hash, profile, settings, metrics, removalCounts }) {
  const payload = {
    tool: ABLATION_VERSION,
    mode: "generative-prior ablation",
    file: fileName,
    input_sha256: hash,
    processed_at: new Date().toISOString(),
    local_pass: profile,
    scaffold_pass: scaffoldModeName(settings.outputMode),
    settings,
    words: {
      original: metrics.originalWords,
      kept: metrics.keptWords,
      stripped: metrics.strippedWords,
    },
    removals: {
      local: removalCounts.local,
      scaffold: removalCounts.scaffold,
      total: metrics.removedFragments,
    },
  };
  return `---\nablation_lab: ${JSON.stringify(payload)}\n---\n\n${text}`;
}

export function downloadText(name, text) {
  try {
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    return true;
  } catch {
    return false;
  }
}
