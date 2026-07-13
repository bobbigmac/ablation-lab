import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  clearBatchHistory,
  countRemovalMethods,
  createBatchJobs,
  downloadText,
  loadBatchHistory,
  prefixAblationFrontmatter,
  saveBatchHistory,
} from "./batch.js";
import { DEFAULT_SETTINGS, PROFILES } from "./refloor.js";
import "./styles.css";

const OUTPUT_OPTIONS = [
  {
    mode: "pattern-off",
    label: "Off",
    slug: null,
    strength: "off",
    title: "Only run the local duplicate-fragment pass.",
  },
  {
    mode: "pattern-ceiling",
    label: "Light",
    slug: "ablation-light",
    strength: "light",
    title: "Add a light scaffold ablation pass that only removes the strongest repeated pattern matches.",
  },
  {
    mode: "pattern-floor2",
    label: "Medium",
    slug: "ablation-medium",
    strength: "medium",
    title: "Add a medium scaffold ablation pass that usually keeps up to two nearby uses of the same pattern.",
  },
  {
    mode: "pattern-floor1",
    label: "Hard",
    slug: "ablation-hard",
    strength: "hard",
    title: "Add a hard scaffold ablation pass that usually keeps one nearby use of the same pattern.",
  },
];

const LOCAL_STRENGTHS = {
  low: "light",
  med: "medium",
  high: "hard",
};

const STORAGE_KEY = "refloor-lab:last-settings";

function loadSavedConfig() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { profile: "med", settings: DEFAULT_SETTINGS };
    const saved = JSON.parse(raw);
    const profile = PROFILES[saved.profile] ? saved.profile : "med";
    const outputMode = OUTPUT_OPTIONS.some((option) => option.mode === saved.settings?.outputMode)
      ? saved.settings.outputMode
      : DEFAULT_SETTINGS.outputMode;
    return {
      profile,
      settings: {
        ...DEFAULT_SETTINGS,
        ...(saved.settings ?? {}),
        outputMode,
      },
    };
  } catch {
    return { profile: "med", settings: DEFAULT_SETTINGS };
  }
}

function renderDiff(text, spans) {
  if (!text) return null;
  const nodes = [];
  let cursor = 0;
  spans.forEach((span, index) => {
    if (span.left > cursor) nodes.push(<span key={`keep-${index}`}>{text.slice(cursor, span.left)}</span>);
    const method = span.method === "scaffold" ? "scaffold" : "local";
    const severity = Math.max(0.18, Math.min(1, span.severity ?? 0.5));
    const title = `${method} pass, severity ${severity.toFixed(2)}: ${span.row.fragment.section}`;
    nodes.push(
      <del
        key={`del-${index}`}
        className={`removed removed-${method}`}
        style={{ "--severity": severity }}
        title={title}
      >
        {text.slice(span.left, span.right)}
      </del>
    );
    cursor = span.right;
  });
  if (cursor < text.length) nodes.push(<span key="tail">{text.slice(cursor)}</span>);
  return nodes;
}

function App() {
  const savedConfig = useRef(loadSavedConfig()).current;
  const [fileName, setFileName] = useState("");
  const [source, setSource] = useState("");
  const [profile, setProfile] = useState(savedConfig.profile);
  const [settings, setSettings] = useState(savedConfig.settings);
  const [result, setResult] = useState(null);
  const [batchMode, setBatchMode] = useState(false);
  const [batchRows, setBatchRows] = useState(loadBatchHistory);
  const [batchStatus, setBatchStatus] = useState("");
  const [isCalculating, setIsCalculating] = useState(false);
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState("");
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const workerRef = useRef(null);
  const runIdRef = useRef(0);
  const dragDepthRef = useRef(0);
  const batchRowsRef = useRef(batchRows);
  const profileRef = useRef(profile);
  const settingsRef = useRef(settings);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ profile, settings }));
    profileRef.current = profile;
    settingsRef.current = settings;
  }, [profile, settings]);

  useEffect(() => {
    batchRowsRef.current = batchRows;
    saveBatchHistory(batchRows);
  }, [batchRows]);

  useEffect(() => {
    workerRef.current = new Worker(new URL("./refloor.worker.js", import.meta.url), { type: "module" });
    workerRef.current.onmessage = (event) => {
      const { id, type, result: nextResult, status: nextStatus, message, progress: nextProgress, fileId, batch } = event.data;
      if (id !== runIdRef.current) return;
      if (type === "batch-status") {
        setBatchStatus(nextStatus);
        return;
      }
      if (type === "batch-progress") {
        if (!fileId) {
          const pct = nextProgress?.progressPct;
          setBatchStatus(pct != null ? `${nextProgress.stage ?? "Building batch prior"} (${Math.round(pct)}%)` : nextProgress?.stage ?? "Building batch prior");
          return;
        }
        setBatchStatus(batch ? `${batch.stage} ${batch.index}/${batch.total}` : nextProgress.stage);
        setBatchRows((rows) => rows.map((row) => row.id === fileId
          ? {
              ...row,
              status: "processing",
              progressStage: nextProgress.stage,
              progressPct: nextProgress.progressPct ?? row.progressPct,
              metrics: { ...row.metrics, ...(nextProgress.metrics ?? {}) },
            }
          : row));
        return;
      }
      if (type === "batch-result") {
        const sourceRow = batchRowsRef.current.find((row) => row.id === fileId);
        const removalCounts = countRemovalMethods(nextResult.spans);
        const refloored = sourceRow ? prefixAblationFrontmatter(nextResult.refloored, {
          fileName: sourceRow.name,
          hash: sourceRow.hash,
          profile: sourceRow.profileSnapshot ?? profileRef.current,
          settings: sourceRow.settingsSnapshot ?? settingsRef.current,
          metrics: nextResult.metrics,
          removalCounts,
        }) : nextResult.refloored;
        const downloaded = sourceRow ? downloadText(sourceRow.name, refloored) : false;
        setBatchRows((rows) => rows.map((row) => row.id === fileId
          ? {
              ...row,
              refloored,
              result: nextResult,
              status: downloaded ? "complete" : "download-failed",
              progressStage: downloaded ? "Downloaded" : "Download blocked",
              progressPct: 100,
              metrics: nextResult.metrics,
              removalCounts,
              downloadStatus: downloaded ? "downloaded" : "failed",
              finishedAt: new Date().toISOString(),
              settingsSnapshot: row.settingsSnapshot ?? settingsRef.current,
              profileSnapshot: row.profileSnapshot ?? profileRef.current,
            }
          : row));
        return;
      }
      if (type === "batch-error") {
        setBatchRows((rows) => rows.map((row) => row.id === fileId
          ? { ...row, status: "error", progressStage: message, error: message, finishedAt: new Date().toISOString() }
          : row));
        return;
      }
      if (type === "batch-complete") {
        setIsCalculating(false);
        setBatchStatus("Batch complete");
        return;
      }
      if (type === "status") {
        setStatus(nextStatus);
        return;
      }
      if (type === "progress") {
        setProgress(nextProgress);
        setStatus(nextProgress.stage);
        return;
      }
      setIsCalculating(false);
      if (type === "error") {
        setError(message);
        return;
      }
      setError("");
      setStatus("");
      setProgress(null);
      setResult(nextResult);
    };
    return () => workerRef.current?.terminate();
  }, []);

  const run = (text = source, nextSettings = settings, nextProfile = profile) => {
    if (!text || !workerRef.current) return;
    const id = runIdRef.current + 1;
    runIdRef.current = id;
    setIsCalculating(true);
    setStatus("Queued");
    setProgress(null);
    setError("");
    workerRef.current.postMessage({ id, text, settings: nextSettings, profile: nextProfile });
  };

  const runBatch = (jobs, nextSettings = settings, nextProfile = profile) => {
    if (!jobs.length || !workerRef.current) return;
    const id = runIdRef.current + 1;
    runIdRef.current = id;
    setIsCalculating(true);
    setBatchStatus("Queued batch");
    setError("");
    workerRef.current.postMessage({
      id,
      type: "batch",
      files: jobs.map((job) => ({ fileId: job.id, name: job.name, text: job.text })),
      settings: nextSettings,
      profile: nextProfile,
    });
  };

  const processFiles = async (files) => {
    if (!files.length) return;
    if (batchMode || files.length > 1) {
      setBatchMode(true);
      const jobs = (await createBatchJobs(files)).map((job) => ({
        ...job,
        settingsSnapshot: settings,
        profileSnapshot: profile,
      }));
      const nextRows = [...jobs, ...batchRowsRef.current];
      batchRowsRef.current = nextRows;
      setBatchRows(nextRows);
      runBatch(jobs);
      return;
    }
    const file = files[0];
    const text = await file.text();
    setFileName(file.name);
    setSource(text);
    run(text);
  };

  const onUpload = async (event) => {
    const files = [...(event.target.files ?? [])];
    event.target.value = "";
    await processFiles(files);
  };

  useEffect(() => {
    const isFileDrag = (event) => Array.from(event.dataTransfer?.types ?? []).includes("Files");

    const onDragEnter = (event) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      dragDepthRef.current += 1;
      setIsDraggingFiles(true);
    };

    const onDragOver = (event) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = isCalculating ? "none" : "copy";
    };

    const onDragLeave = (event) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
      if (dragDepthRef.current === 0) setIsDraggingFiles(false);
    };

    const onDrop = async (event) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      dragDepthRef.current = 0;
      setIsDraggingFiles(false);
      if (isCalculating) return;
      await processFiles([...(event.dataTransfer?.files ?? [])]);
    };

    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [batchMode, isCalculating, profile, settings]);

  const clearHistory = () => {
    clearBatchHistory();
    setBatchRows([]);
  };

  const retryBatchDownload = (row) => {
    if (!row.refloored) return;
    const downloaded = downloadText(row.name, row.refloored);
    setBatchRows((rows) => rows.map((item) => item.id === row.id
      ? {
          ...item,
          status: downloaded ? "complete" : "download-failed",
          progressStage: downloaded ? "Downloaded" : "Download blocked",
          downloadStatus: downloaded ? "downloaded" : "failed",
        }
      : item));
  };

  const activeOutput = OUTPUT_OPTIONS.find((option) => option.mode === settings.outputMode) ?? OUTPUT_OPTIONS[0];
  const outputLabel = activeOutput.slug ? `${profile}.${activeOutput.slug}` : profile;
  const exportName = fileName.replace(/\.md$/i, `.${outputLabel}.md`) || `refloored.${outputLabel}.md`;
  const liveMetrics = progress?.metrics;
  const progressPct = progress?.progressPct ?? 0;
  const removalCounts = (result?.spans ?? []).reduce((counts, span) => {
    const method = span.method === "scaffold" ? "scaffold" : "local";
    counts[method] += 1;
    return counts;
  }, { local: 0, scaffold: 0 });

  return (
    <main>
      {isDraggingFiles ? (
        <div className="drop-overlay">
          <div>
            <strong>{isCalculating ? "Calculation running" : "Drop *.md file(s)"}</strong>
            <span>{isCalculating ? "Finish the current run before adding files." : "One file opens the single-file lab; multiple files append to batch mode."}</span>
          </div>
        </div>
      ) : null}
      <aside>
        <h1>Ablation Lab</h1>
        <label className="upload">
          <input type="file" multiple disabled={isCalculating} accept=".md,text/markdown,text/plain" onChange={onUpload} />
          <span>Choose *.md file(s)</span>
        </label>

        <section>
          <h2>Scaffold Pass</h2>
          <div className="segmented">
            {OUTPUT_OPTIONS.map((option) => (
              <button
                key={option.mode}
                className={`pass-button scaffold-button strength-${option.strength}${settings.outputMode === option.mode ? " active" : ""}`}
                title={option.title}
                onClick={() => setSettings((s) => ({ ...s, outputMode: option.mode }))}
              >
                {option.label}
              </button>
            ))}
          </div>
        </section>

        <section>
          <h2>Local Pass</h2>
          <div className="segmented">
            {Object.keys(PROFILES).map((name) => (
              <button
                key={name}
                className={`pass-button local-button strength-${LOCAL_STRENGTHS[name] ?? "medium"}${profile === name ? " active" : ""}`}
                onClick={() => setProfile(name)}
              >
                {name}
              </button>
            ))}
          </div>
        </section>

        <section className="controls">
          <h2>Search</h2>
          <Control
            label="Radius"
            value={settings.paragraphRadius}
            tooltip="How many nearby paragraphs each fragment can compare against. Higher values catch more distant repetition but cost more time."
            onDown={() => setSettings((s) => ({ ...s, paragraphRadius: Math.max(1, s.paragraphRadius - 2) }))}
            onUp={() => setSettings((s) => ({ ...s, paragraphRadius: s.paragraphRadius + 2 }))}
          />
          <Control
            label="Match"
            value={settings.minMatchScore.toFixed(2)}
            tooltip="The minimum token-overlap score for ordinary local repetition candidates. Lower values flag looser matches."
            onDown={() => setSettings((s) => ({ ...s, minMatchScore: Math.max(0.05, Number((s.minMatchScore - 0.02).toFixed(2))) }))}
            onUp={() => setSettings((s) => ({ ...s, minMatchScore: Math.min(0.8, Number((s.minMatchScore + 0.02).toFixed(2))) }))}
          />
          <Control
            label="Ablate"
            value={settings.patternMinScore.toFixed(2)}
            tooltip="The minimum score for matching the diffuse scaffold pattern model. Lower values let ablation remove weaker pattern matches."
            onDown={() => setSettings((s) => ({ ...s, patternMinScore: Math.max(0.2, Number((s.patternMinScore - 0.02).toFixed(2))) }))}
            onUp={() => setSettings((s) => ({ ...s, patternMinScore: Math.min(0.9, Number((s.patternMinScore + 0.02).toFixed(2))) }))}
          />
          <Control
            label="Min tokens"
            value={settings.minTokens}
            tooltip="The fewest useful tokens a fragment must have before it can be considered. Higher values ignore very short fragments."
            onDown={() => setSettings((s) => ({ ...s, minTokens: Math.max(1, s.minTokens - 1) }))}
            onUp={() => setSettings((s) => ({ ...s, minTokens: s.minTokens + 1 }))}
          />
          <Control
            label="Posting cap"
            value={settings.maxCandidatesPerToken}
            tooltip="The maximum candidate fragments kept per shared token. Lower values run faster; higher values search more exhaustively."
            onDown={() => setSettings((s) => ({ ...s, maxCandidatesPerToken: Math.max(20, s.maxCandidatesPerToken - 20) }))}
            onUp={() => setSettings((s) => ({ ...s, maxCandidatesPerToken: s.maxCandidatesPerToken + 20 }))}
          />
          <Control
            label="Growth"
            value={settings.frontierGrowth === 0 ? "auto" : settings.frontierGrowth}
            tooltip="How much the search radius grows when repeated patterns are found. 0/auto uses the profile default (low=1, med=3, high=4). 1 = fixed radius (no growth). Higher = reach further across the document for heavily repeating patterns."
            onDown={() => setSettings((s) => ({ ...s, frontierGrowth: Math.max(0, s.frontierGrowth - 1) }))}
            onUp={() => setSettings((s) => ({ ...s, frontierGrowth: Math.min(8, s.frontierGrowth + 1) }))}
          />
        </section>

        <button className="primary" disabled={batchMode || !source || isCalculating} onClick={() => run()}>
          {isCalculating ? "Calculating..." : "Run"}
        </button>
        <button disabled={batchMode || !result || isCalculating} onClick={() => downloadText(exportName, result.refloored)}>
          Export markdown
        </button>
        {!batchMode ? <button disabled={isCalculating} onClick={() => setBatchMode(true)}>Batch Mode</button> : null}
        {batchMode ? <button disabled={!batchRows.length || isCalculating} onClick={clearHistory}>Clear Batch History</button> : null}
        <button className="help-link" onClick={() => setHelpOpen(true)}>Help &amp; Documentation</button>
      </aside>

      <section className="workspace">
        {batchMode ? (
          <BatchWorkspace
            rows={batchRows}
            batchStatus={batchStatus}
            isCalculating={isCalculating}
            onRetryDownload={retryBatchDownload}
          />
        ) : (
          <>
          <div className="metrics">
          {isCalculating ? (
            <>
              <Metric label="Source Words" value={liveMetrics?.originalWords ?? 0} />
              <Metric label="Words Indexed" value={liveMetrics?.processedWords ?? 0} />
              <Metric label="Fragments" value={liveMetrics?.processedFragments ?? 0} />
              <Metric label="Stage %" value={progressPct.toFixed(0)} />
              <Metric label="Staged Cuts" value={liveMetrics?.stagedRemovals ?? 0} />
              <Metric label="Pattern Hits" value={liveMetrics?.patternCandidates ?? 0} />
            </>
          ) : (
            <>
              <Metric label="Original" value={result?.metrics.originalWords ?? 0} />
              <Metric label="Kept" value={result?.metrics.keptWords ?? 0} />
              <Metric label="Stripped" value={result?.metrics.strippedWords ?? 0} />
              <Metric label="Kept %" value={result ? result.metrics.keptPct.toFixed(1) : "0.0"} />
              <Metric label="Removed" value={result?.metrics.removedFragments ?? 0} />
              <Metric label="Pattern Hits" value={result?.patternCandidates.length ?? 0} />
            </>
          )}
          </div>

          {result?.timings ? <TimingStrip timings={result.timings} /> : null}
          {error ? <div className="error">{error}</div> : null}

          <details className="diagnostics" open={diagnosticsOpen} onToggle={(event) => setDiagnosticsOpen(event.currentTarget.open)}>
          <summary>Scaffold templates, short phrases, and inside-fragment flags</summary>
          <div className="diagnostics-grid">
            <section>
              <h2>Diffuse Scaffold Templates</h2>
              {result?.patternModel ? <p className="note">Ablation model: <b>{result.patternModel.softText}</b></p> : null}
              <ol>
                {(result?.templateRows ?? []).slice(0, 40).map((row) => (
                  <li key={row.softText}>
                    <strong>{row.softText}</strong>
                    <span>{row.count} hits, {row.positionVariety} positional variants, {row.sectionCount} sections</span>
                    <p>{row.positions.map((position, index) => `${index + 1}: ${position.map(([token, count]) => `${token} (${count})`).join(", ")}`).join("; ")}</p>
                  </li>
                ))}
              </ol>
              <h2>Short Phrase Diagnostics</h2>
              <div className="chips">
                {(result?.phraseRows ?? []).slice(0, 48).map(([phrase, count]) => (
                  <span key={phrase}>{phrase} <b>{count}</b></span>
                ))}
              </div>
            </section>
            <section>
              <h2>Inside-Fragment Flags</h2>
              <ol>
                {(result?.patternRows ?? []).slice(0, 12).map((row, index) => (
                  <li key={`${row.fragment.idx}-${index}`}>
                    <strong>{row.fragment.section}</strong>
                    <span>{row.reasons.join(", ")}</span>
                    <p>{row.fragment.text}</p>
                  </li>
                ))}
              </ol>
            </section>
          </div>
          </details>

          <section className="panel diff">
          <div className="diff-head">
            <h2>Inline Removal Diff</h2>
            <div className="legend" aria-label="Removal colour legend">
              <span><i className="swatch swatch-local" />Local{result ? <b>-{removalCounts.local}</b> : null}</span>
              <span><i className="swatch swatch-scaffold" />Scaffold{result ? <b>-{removalCounts.scaffold}</b> : null}</span>
            </div>
            {isCalculating ? <span className="busy"><i className="spinner" aria-hidden="true" />{status || "Calculating"} ({progressPct.toFixed(0)}%)</span> : null}
          </div>
          <div className="diff-body">
            <pre>{renderDiff(source, result?.spans ?? [])}</pre>
          </div>
          </section>
          </>
        )}
      </section>
      <footer className="footer">
        <a href="https://github.com/bobbigmac/ablation-lab" target="_blank" rel="noopener noreferrer">GitHub</a>
      </footer>
      {helpOpen ? <HelpModal onClose={() => setHelpOpen(false)} /> : null}
    </main>
  );
}

function HelpModal({ onClose }) {
  return (
    <div className="help-overlay" onClick={onClose}>
      <div className="help-modal" onClick={(e) => e.stopPropagation()}>
        <div className="help-head">
          <h2>Ablation Lab — Help</h2>
          <button className="help-close" onClick={onClose} aria-label="Close help">&times;</button>
        </div>
        <div className="help-body">
          <section>
            <h3>What it does</h3>
            <p>Ablation Lab detects and removes repetitive text patterns in AI-generated prose. It splits your markdown into thought-unit fragments, tokenises them, and scores fragments by how many nearby fragments share meaningful token overlap. Redundant fragments are excised with responsible punctuation healing, and diffuse scaffold patterns (repeated structural templates with variable slots) are detected and optionally ablated.</p>
          </section>
          <section>
            <h3>Protected names (<code>characters:</code> field)</h3>
            <p>If your markdown file has YAML frontmatter with a <code>characters:</code> field, the tokens in those names are excluded from the repetition index. This prevents repeated proper names — people, places, entities — from inflating repetition scores.</p>
            <p><strong>Format:</strong> Add a frontmatter block at the top of your file:</p>
            <pre className="help-code">{`---
characters: [Alice, Bob, Dr. Chen, Alexandria, Egypt]
---

# Your Title`}</pre>
            <p>Each name is tokenised and non-stopword tokens are added to the exclusion set. This is per-document — different files can have different character sets.</p>
            <p>Character names are also auto-extracted from <strong>speaker labels</strong> (lines starting with <code>Name:</code>) in the document body.</p>
          </section>
          <section>
            <h3>@ markers in output</h3>
            <p>When a fragment is removed, an <code>@</code> marker is inserted in its place. These markers indicate where text was excised and can be used as edit points for manual or LLM-based revision. Multiple adjacent markers are consolidated, and surrounding punctuation is cleaned up automatically.</p>
            <p>In the exported markdown, <code>@</code> marks every spot where the tool removed content. You can search for <code>@</code> in your editor to review each cut.</p>
          </section>
          <section>
            <h3>Profiles</h3>
            <p>The <strong>Local Pass</strong> has three profiles controlling how aggressively fragments are removed:</p>
            <ul>
              <li><strong>low</strong> — conservative; requires high badness, protects capitalised fragments, no spatial growth</li>
              <li><strong>med</strong> — balanced; moderate thresholds, allows up to 3× radius growth</li>
              <li><strong>high</strong> — aggressive; low thresholds, up to 4× radius growth, removes up to 35% of fragments</li>
            </ul>
          </section>
          <section>
            <h3>Scaffold Pass</h3>
            <p>The <strong>Scaffold Pass</strong> detects repeated structural templates with variable slots — patterns like <code>[token] was the thing that [token]</code> where fixed positions recur but wildcard slots vary. Four levels are available: Off, Light, Medium, and Hard.</p>
          </section>
          <section>
            <h3>Search controls</h3>
            <ul>
              <li><strong>Radius</strong> — How many nearby paragraphs each fragment compares against</li>
              <li><strong>Match</strong> — Minimum token-overlap score for candidates</li>
              <li><strong>Ablate</strong> — Minimum score for scaffold pattern matching</li>
              <li><strong>Min tokens</strong> — Fewest tokens a fragment needs to be considered</li>
              <li><strong>Posting cap</strong> — Max candidates per shared token (performance control)</li>
              <li><strong>Growth</strong> — How far the search radius expands for heavily repeating patterns (0 = auto)</li>
            </ul>
          </section>
          <section>
            <h3>Batch mode</h3>
            <p>Upload multiple <code>.md</code> files to process them all at once. Results are auto-downloaded with ablation frontmatter prefixed. Batch history (metadata only, no file contents) is stored in localStorage for reference.</p>
          </section>
          <section>
            <h3>Privacy</h3>
            <p>Everything runs in your browser. No server, no logging, no telemetry. File contents never leave your machine.</p>
          </section>
        </div>
      </div>
    </div>
  );
}

function BatchWorkspace({ rows, batchStatus, isCalculating, onRetryDownload }) {
  const totals = rows.reduce((out, row) => {
    out.files += 1;
    out.kept += row.metrics?.keptWords ?? 0;
    out.stripped += row.metrics?.strippedWords ?? 0;
    out.local += row.removalCounts?.local ?? 0;
    out.scaffold += row.removalCounts?.scaffold ?? 0;
    if (row.status === "complete") out.complete += 1;
    if (row.status === "error" || row.status === "download-failed") out.failed += 1;
    return out;
  }, { files: 0, complete: 0, failed: 0, kept: 0, stripped: 0, local: 0, scaffold: 0 });

  return (
    <section className="batch-workspace">
      <div className="batch-head">
        <div>
          <h2>Batch Ablation</h2>
          <p>{batchStatus || "Upload multiple files to run client-side batch ablation."}</p>
        </div>
        {isCalculating ? <span className="busy"><i className="spinner" aria-hidden="true" />{batchStatus || "Processing batch"}</span> : null}
      </div>

      <div className="metrics">
        <Metric label="Files" value={totals.files} />
        <Metric label="Complete" value={totals.complete} />
        <Metric label="Failed" value={totals.failed} />
        <Metric label="Stripped" value={totals.stripped} />
        <Metric label="Local Cuts" value={`-${totals.local}`} />
        <Metric label="Scaffold Cuts" value={`-${totals.scaffold}`} />
      </div>

      <div className="batch-list">
        {rows.length ? rows.map((row) => (
          <article key={row.id} className={`batch-row status-${row.status}`}>
            <header>
              <div>
                <h3>{row.name}</h3>
                <p><span>sha256</span> <code title={row.hash}>{row.hash?.slice(0, 16) ?? "unknown"}</code></p>
              </div>
              <strong>{row.status}</strong>
            </header>
            <div className="batch-progress">
              <span style={{ width: `${Math.max(0, Math.min(100, row.progressPct ?? 0))}%` }} />
            </div>
            <div className="batch-stats">
              <span>{row.progressStage || "Queued"}</span>
              <span>{Math.round(row.progressPct ?? 0)}%</span>
              <span>kept {row.metrics?.keptWords ?? 0}</span>
              <span>stripped {row.metrics?.strippedWords ?? 0}</span>
              <span className="local-count">local -{row.removalCounts?.local ?? 0}</span>
              <span className="scaffold-count">scaffold -{row.removalCounts?.scaffold ?? 0}</span>
              <span>download {row.downloadStatus ?? "pending"}</span>
            </div>
            {row.status === "download-failed" && row.refloored ? (
              <button onClick={() => onRetryDownload(row)}>Retry download</button>
            ) : null}
            {row.error ? <p className="row-error">{row.error}</p> : null}
          </article>
        )) : <div className="empty-state">No batch history yet.</div>}
      </div>
    </section>
  );
}

function Control({ label, value, tooltip, onDown, onUp }) {
  return (
    <div className="control" title={tooltip} data-tooltip={tooltip}>
      <span className="control-label">{label}</span>
      <button onClick={onDown} aria-label={`Decrease ${label}`}>-</button>
      <output>{value}</output>
      <button onClick={onUp} aria-label={`Increase ${label}`}>+</button>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TimingStrip({ timings }) {
  const rows = ["parse", "fragment", "score", "select", "diagnostics", "pattern", "apply", "total"];
  return (
    <div className="timings">
      {rows.map((name) => (
        <span key={name}>{name} <b>{Math.round(timings[name] ?? 0)}ms</b></span>
      ))}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
