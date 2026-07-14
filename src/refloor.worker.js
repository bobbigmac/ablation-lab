import { analyze, buildBatchPatternModel, prepareDocumentForBatch } from "./refloor.js";

self.onmessage = (event) => {
  const { id, type = "single", text, settings, profile, files = [] } = event.data;
  let lastProgressAt = 0;
  const postProgress = (progress) => {
    const now = globalThis.performance?.now ? globalThis.performance.now() : Date.now();
    if (!progress.force && now - lastProgressAt < 1000) return;
    lastProgressAt = now;
    self.postMessage({ id, type: "progress", progress });
  };
  try {
    if (type === "batch") {
      self.postMessage({ id, type: "batch-status", status: "Indexing batch prior" });
      const preparedDocs = files.map((file, i) => {
        self.postMessage({ id, type: "batch-progress", progress: { stage: `Indexing files (${i + 1}/${files.length})`, progressPct: (i / files.length) * 5, force: true } });
        return prepareDocumentForBatch(file.text, settings);
      });
      let lastPriorProgressAt = 0;
      const { patternModels } = buildBatchPatternModel(preparedDocs, (progress) => {
        const now = globalThis.performance?.now ? globalThis.performance.now() : Date.now();
        if (!progress.force && now - lastPriorProgressAt < 1000) return;
        lastPriorProgressAt = now;
        self.postMessage({ id, type: "batch-progress", progress: { ...progress, stage: progress.stage ?? "Building shared scaffold prior" } });
      });
      files.forEach((file, index) => {
        let lastFileProgressAt = 0;
        const postFileProgress = (progress) => {
          const now = globalThis.performance?.now ? globalThis.performance.now() : Date.now();
          if (!progress.force && now - lastFileProgressAt < 1000) return;
          lastFileProgressAt = now;
          self.postMessage({
            id,
            type: "batch-progress",
            fileId: file.fileId,
            progress,
            batch: { index: index + 1, total: files.length, stage: "Processing files" },
          });
        };
        try {
          self.postMessage({
            id,
            type: "batch-progress",
            fileId: file.fileId,
            progress: { stage: "Starting", progressPct: 0, metrics: {}, force: true },
            batch: { index: index + 1, total: files.length, stage: "Processing files" },
          });
          const result = analyze(file.text, settings, profile, postFileProgress, { patternModels });
          self.postMessage({ id, type: "batch-result", fileId: file.fileId, result });
        } catch (error) {
          self.postMessage({
            id,
            type: "batch-error",
            fileId: file.fileId,
            message: error instanceof Error ? error.message : String(error),
          });
        }
      });
      self.postMessage({ id, type: "batch-complete" });
      return;
    }

    self.postMessage({ id, type: "status", status: "Calculating" });
    const result = analyze(text, settings, profile, postProgress);
    self.postMessage({ id, type: "result", result });
  } catch (error) {
    self.postMessage({
      id,
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
