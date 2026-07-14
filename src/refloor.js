const STOPWORDS = new Set([
  "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of",
  "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
  "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
  "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those",
]);

export const PROFILES = {
  low: {
    minBadness: 2.8,
    minViable: 0.5,
    minMatches: 2,
    minReuse: 0.65,
    maxUnique: 2,
    maxRemoveRatio: 0.08,
    allowCapitalized: false,
    frontierGrowth: 1,
  },
  med: {
    minBadness: 1.8,
    minViable: 0.2,
    minMatches: 1,
    minReuse: 0.5,
    maxUnique: 4,
    maxRemoveRatio: 0.18,
    allowCapitalized: true,
    frontierGrowth: 3,
  },
  high: {
    minBadness: 1.0,
    minViable: 0.1,
    minMatches: 1,
    minReuse: 0.4,
    maxUnique: 6,
    maxRemoveRatio: 0.35,
    allowCapitalized: true,
    frontierGrowth: 4,
  },
};

export const DEFAULT_SETTINGS = {
  paragraphRadius: 12,
  frontierGrowth: 0,
  minMatchScore: 0.2,
  minTokens: 3,
  maxCandidatesPerToken: 180,
  top: 30,
  outputMode: "pattern-threshold-dedup",
  patternMinScore: 0.52,
};

function stripFrontmatter(text) {
  if (!text.startsWith("---\n")) return ["", text, 0];
  const end = text.indexOf("\n---\n", 4);
  if (end === -1) return ["", text, 0];
  const offset = end + 5;
  return [text.slice(0, offset), text.slice(offset), offset];
}

function normalizeToken(raw) {
  const trimmed = raw.toLowerCase().replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, "");
  return /[a-z0-9]/.test(trimmed) ? trimmed : null;
}

function tokenize(text, excluded = new Set()) {
  const out = [];
  for (const raw of text.match(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g) ?? []) {
    const token = normalizeToken(raw);
    if (token && !excluded.has(token)) out.push(token);
  }
  return out;
}

function extractCharacterTokens(frontmatter) {
  const tokens = new Set();
  for (const line of frontmatter.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("characters:")) continue;
    const payload = trimmed.split(":").slice(1).join(":").trim();
    const names = payload.startsWith("[") && payload.endsWith("]")
      ? payload.slice(1, -1).split(",")
      : payload.split(",");
    for (const name of names) {
      for (const token of tokenize(name.replace(/["']/g, ""))) {
        if (!STOPWORDS.has(token)) tokens.add(token);
      }
    }
  }
  return tokens;
}

function extractSpeakerTokens(text) {
  const tokens = new Set();
  for (const line of text.split(/\r?\n/)) {
    const match = line.trim().match(/^([A-Za-z][A-Za-z0-9 _.'’-]{0,48}):\s+/);
    if (!match) continue;
    for (const token of tokenize(match[1].replace(/["']/g, ""))) {
      if (!STOPWORDS.has(token)) tokens.add(token);
    }
  }
  return tokens;
}

function isSpeakerColon(text, colonIndex) {
  const lineStart = Math.max(text.lastIndexOf("\n", colonIndex - 1) + 1, 0);
  const label = text.slice(lineStart, colonIndex).trim();
  return /^[A-Za-z][A-Za-z0-9 _.'’-]{0,48}$/.test(label);
}

function cleanFragment(text) {
  return text
    .trim()
    .replace(/^[\s"'“”‘’*_(){}[\]<>-]+/g, "")
    .replace(/[\s"'“”‘’*_(){}[\]<>-]+$/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function firstAlpha(text) {
  return text.match(/[A-Za-z]/)?.[0] ?? "";
}

function parseParagraphs(text) {
  const [frontmatter, body, bodyOffset] = stripFrontmatter(text);
  const paragraphs = [];
  const lines = body.split(/\r?\n/);
  let section = "(preamble)";
  let current = [];
  let currentStartLine = 1;
  let currentStartOffset = bodyOffset;
  let offset = bodyOffset;
  let paragraphIndex = 0;

  function flush(endLine) {
    const raw = current.join("\n").trim();
    if (!raw) {
      current = [];
      return;
    }
    paragraphs.push({
      section,
      paraIndex: paragraphIndex++,
      startLine: currentStartLine,
      endLine,
      text: raw,
      sourceStart: currentStartOffset,
      sourceEnd: currentStartOffset + raw.length,
    });
    current = [];
  }

  lines.forEach((line, index) => {
    const lineNo = index + 1;
    const stripped = line.trim();
    if (/^#{1,6}\s+/.test(stripped)) {
      flush(lineNo - 1);
      section = stripped.replace(/^#{1,6}\s+/, "").trim() || section;
    } else if (stripped === "---" || !stripped) {
      flush(lineNo - 1);
    } else {
      if (!current.length) {
        currentStartLine = lineNo;
        currentStartOffset = offset + line.search(/\S/);
      }
      current.push(stripped);
    }
    offset += line.length + 1;
  });
  flush(lines.length);
  return { frontmatter, paragraphs };
}

function splitFragments(paragraph, excluded, minTokens) {
  const fragments = [];
  for (const match of paragraph.text.matchAll(/[^,.;:!?—–]+/g)) {
    const raw = match[0];
    const text = cleanFragment(raw);
    if (/^[A-Za-z][A-Za-z0-9 _.'’-]{0,48}$/.test(text) && paragraph.text.slice(match.index + raw.length, match.index + raw.length + 1) === ":") continue;
    const tokens = tokenize(text, excluded);
    if (tokens.length < minTokens) continue;
    const sourceStart = paragraph.sourceStart + match.index + raw.indexOf(text);
    const sourceEnd = sourceStart + text.length;
    const alpha = firstAlpha(text);
    fragments.push({
      section: paragraph.section,
      paraIndex: paragraph.paraIndex,
      startLine: paragraph.startLine,
      text,
      tokens,
      contentTokens: tokens,
      startsCapitalized: alpha && alpha === alpha.toUpperCase(),
      sourceStart,
      sourceEnd,
    });
  }
  return fragments;
}

function ngrams(tokens, n) {
  const out = [];
  for (let i = 0; i <= tokens.length - n; i += 1) out.push(tokens.slice(i, i + n));
  return out;
}

function key(window) {
  return window.join("\u0001");
}

function jaccard(a, b) {
  if (!a.size || !b.size) return 0;
  let both = 0;
  for (const item of a) if (b.has(item)) both += 1;
  return both / (a.size + b.size - both);
}

function sequenceRatio(a, b) {
  if (!a.length || !b.length) return 0;
  const dp = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let i = 1; i <= a.length; i += 1) {
    for (let j = 1; j <= b.length; j += 1) {
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  return (2 * dp[a.length][b.length]) / (a.length + b.length);
}

function similarity(a, b) {
  const rawOverlap = jaccard(new Set(a.tokens), new Set(b.tokens));
  const rawSeq = sequenceRatio(a.tokens, b.tokens);
  const bigram = jaccard(new Set(ngrams(a.tokens, 2).map(key)), new Set(ngrams(b.tokens, 2).map(key)));
  const trigram = jaccard(new Set(ngrams(a.tokens, 3).map(key)), new Set(ngrams(b.tokens, 3).map(key)));
  const size = 1 - Math.abs(a.tokens.length - b.tokens.length) / Math.max(a.tokens.length, b.tokens.length);
  const shared = [...new Set(a.tokens)].filter((token) => new Set(b.tokens).has(token)).length;
  const exchange = ((a.tokens.includes("i") && b.tokens.includes("you")) || (a.tokens.includes("you") && b.tokens.includes("i"))) && rawOverlap >= 0.35;
  const score = (0.38 * rawOverlap) + (0.2 * rawSeq) + (0.16 * bigram) + (0.08 * trigram) + (0.18 * size);
  return { score, rawOverlap, rawSeq, bigram, trigram, shared, exchange };
}

function buildIndex(fragments) {
  const index = new Map();
  fragments.forEach((frag, idx) => {
    frag.idx = idx;
    for (const token of new Set(frag.tokens)) {
      if (!index.has(token)) index.set(token, []);
      index.get(token).push(idx);
    }
  });
  return index;
}

function candidateIds(frag, fragments, index, tokens, settings, effectiveRadius) {
  const ids = new Set();
  for (const token of new Set(tokens)) {
    const posting = index.get(token) ?? [];
    if (posting.length > settings.maxCandidatesPerToken) continue;
    for (const idx of posting) {
      if (idx === frag.idx) continue;
      const other = fragments[idx];
      if (Math.abs(other.paraIndex - frag.paraIndex) <= effectiveRadius) ids.add(idx);
    }
  }
  return ids;
}

function keepHint(frag, matches, fragments) {
  const cluster = [frag, ...matches.map((m) => fragments[m.idx])];
  const earliest = cluster.reduce((a, b) => (a.idx < b.idx ? a : b));
  const caps = cluster.filter((item) => item.startsCapitalized);
  const earliestCap = caps.reduce((a, b) => (!a || b.idx < a.idx ? b : a), null);
  if (frag.idx === earliest.idx) return "earliest";
  if (earliestCap && frag.idx === earliestCap.idx) return "earliest-capitalized";
  return "redundant-candidate";
}

function scoreFragments(fragments, settings, report) {
  const index = buildIndex(fragments);
  const rows = [];
  let processedWords = 0;
  const growthMultiplier = settings.frontierGrowth ?? 1;
  for (const [position, frag] of fragments.entries()) {
    processedWords += frag.tokens.length;
    const matches = [];
    const matched = new Set();
    let frontierTokens = new Set(frag.tokens);
    let frontierBoost = 1;

    function collect(ids) {
      for (const idx of ids) {
        if (matched.has(idx)) continue;
        const other = fragments[idx];
        const metrics = similarity(frag, other);
        if (metrics.shared < 1 && metrics.bigram === 0) continue;
        if (metrics.score < settings.minMatchScore) continue;
        const adjusted = Math.max(0, metrics.score - (metrics.exchange ? 0.12 : 0));
        const distance = Math.abs(other.paraIndex - frag.paraIndex);
        matches.push({ idx, metrics, distance, weighted: adjusted / (1 + distance) });
        matched.add(idx);
      }
    }

    const effectiveRadius = settings.paragraphRadius * Math.min(frontierBoost, growthMultiplier);
    collect(candidateIds(frag, fragments, index, frag.tokens, settings, effectiveRadius));
    matches.sort((a, b) => b.weighted - a.weighted || a.distance - b.distance || a.idx - b.idx);

    while (matches.length) {
      const bridge = [];
      const seen = new Set(frontierTokens);
      for (const match of matches.slice(0, Math.min(matches.length, 4 * frontierBoost))) {
        for (const token of fragments[match.idx].tokens) {
          if (!seen.has(token)) {
            seen.add(token);
            bridge.push(token);
          }
        }
        if (bridge.length >= 4 * frontierBoost) break;
      }
      if (!bridge.length) break;
      const expandRadius = settings.paragraphRadius * Math.min(frontierBoost, growthMultiplier);
      const expanded = candidateIds(frag, fragments, index, bridge, settings, expandRadius);
      for (const id of matched) expanded.delete(id);
      if (!expanded.size) break;
      frontierTokens = new Set([...frontierTokens, ...bridge]);
      frontierBoost *= 2;
      collect(expanded);
      matches.sort((a, b) => b.weighted - a.weighted || a.distance - b.distance || a.idx - b.idx);
    }

    if (!matches.length) continue;
    const counts = new Map();
    for (const match of matches) {
      for (const token of fragments[match.idx].tokens) counts.set(token, (counts.get(token) ?? 0) + 1);
    }
    const uniqueTokens = new Set(frag.tokens);
    const reusable = [...uniqueTokens].filter((token) => counts.has(token)).length;
    const unique = uniqueTokens.size - reusable;
    const reusableRatio = reusable / Math.max(1, uniqueTokens.size);
    const density = matches.reduce((sum, item) => sum + item.weighted, 0);
    const badness = density * (1 + Math.min(matches.length, 8) / 4) * (0.5 + reusableRatio);
    rows.push({
      fragment: frag,
      matches,
      matchCount: matches.length,
      density,
      badness,
      reusable,
      unique,
      reusableRatio,
    });
    report?.({
      stage: "Scoring local repetition",
      stagePct: 25 + (position / Math.max(1, fragments.length)) * 30,
      metrics: { processedFragments: position + 1, processedWords, nearMatches: rows.length },
    });
  }
  rows.forEach((row) => { row.hint = keepHint(row.fragment, row.matches, fragments); });
  return rows.sort((a, b) => b.badness - a.badness || b.matchCount - a.matchCount || b.fragment.idx - a.fragment.idx);
}

function repeatedPhrases(rows) {
  const counts = new Map();
  for (const row of rows) {
    for (const n of [2, 3, 4]) {
      for (const phrase of ngrams(row.fragment.tokens, n).map((window) => window.join(" "))) {
        counts.set(phrase, (counts.get(phrase) ?? 0) + 1);
      }
    }
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).filter(([, count]) => count >= 4).slice(0, 40);
}

function wildcardSets(length) {
  const out = [];
  const maxWildcards = Math.min(3, Math.max(1, length - 4));
  for (let wildcardCount = 1; wildcardCount <= maxWildcards; wildcardCount += 1) {
    const combo = [];
    function choose(start) {
      if (combo.length === wildcardCount) {
        out.push([...combo]);
        return;
      }
      for (let pos = start; pos < length; pos += 1) {
        combo.push(pos);
        choose(pos + 1);
        combo.pop();
      }
    }
    choose(1);
  }
  return out;
}

function scanTemplatesForFragments(fragments, rows, wildcardCache, hotTokens, report) {
  const useFilter = hotTokens && hotTokens.size > 0;
  let processedWords = 0;
  let pruneCounter = 0;
  for (const [position, frag] of fragments.entries()) {
    processedWords += frag.tokens.length;
    for (let length = 5; length <= 9; length += 1) {
      if (frag.tokens.length < length) continue;
      for (let start = 0; start <= frag.tokens.length - length; start += 1) {
        const window = frag.tokens.slice(start, start + length);
        if (useFilter) {
          let hasHot = false;
          for (const tok of window) {
            if (hotTokens.has(tok)) { hasHot = true; break; }
          }
          if (!hasHot) continue;
        }
        for (const wildcards of wildcardCache.get(length)) {
          const template = window.map((token, index) => (wildcards.includes(index) ? "_" : token));
          if (new Set(template.filter((token) => token !== "_")).size < 3) continue;
          const id = template.join("\u0001");
          if (!rows.has(id)) {
            rows.set(id, {
              template,
              text: template.join(" "),
              count: 0,
              slots: wildcards.map(() => new Map()),
              positions: template.map(() => new Map()),
              occurrences: new Set(),
              footprint: new Set(),
              examples: [],
              sections: new Map(),
            });
          }
          const row = rows.get(id);
          const occurrence = `${frag.idx}:${start}`;
          if (row.occurrences.has(occurrence)) continue;
          row.occurrences.add(occurrence);
          for (let pos = 0; pos < length; pos += 1) row.footprint.add(`${frag.idx}:${start + pos}`);
          row.count += 1;
          row.sections.set(frag.section, (row.sections.get(frag.section) ?? 0) + 1);
          window.forEach((token, position) => {
            const counter = row.positions[position];
            counter.set(token, (counter.get(token) ?? 0) + 1);
          });
          wildcards.forEach((pos, slotIndex) => {
            const slot = row.slots[slotIndex];
            slot.set(window[pos], (slot.get(window[pos]) ?? 0) + 1);
          });
          const example = window.join(" ");
          if (row.examples.length < 4 && !row.examples.includes(example)) row.examples.push(example);
        }
      }
    }
    pruneCounter += 1;
    if (pruneCounter >= 500) {
      pruneCounter = 0;
      for (const [id, row] of rows) {
        if (row.count < 2) rows.delete(id);
      }
    }
    report?.({
      stage: "Finding scaffold templates",
      stagePct: 58 + (position / Math.max(1, fragments.length)) * 18,
      metrics: { processedFragments: position + 1, processedWords, templateSeeds: rows.size },
    });
  }
}

function extractHotTokens(rows, minTemplateCount = 2) {
  const tokenFreq = new Map();
  for (const row of rows.values()) {
    if (row.count < minTemplateCount) continue;
    for (const token of row.template) {
      if (token === "_") continue;
      tokenFreq.set(token, (tokenFreq.get(token) ?? 0) + 1);
    }
  }
  const hot = new Set();
  for (const [token, templateCount] of tokenFreq) {
    if (templateCount >= 2) hot.add(token);
  }
  return hot;
}

function rankAndDedupTemplates(rows) {
  for (const row of rows.values()) {
    row.occurrences = null;
  }
  const ranked = [...rows.values()].map((row) => {
    const slotVariety = row.slots.reduce((sum, slot) => sum + slot.size, 0);
    const positionVariety = row.positions.reduce((sum, position) => sum + position.size, 0);
    const fixedCount = row.template.filter((token) => token !== "_").length;
    const score = row.count
      * (1 + Math.min(slotVariety, 24) / 8)
      * (1 + row.slots.length / 3)
      * (fixedCount / row.template.length)
      * (1 + Math.min(row.sections.size, 4) / 8);
    return {
      ...row,
      slotVariety,
      positionVariety,
      softText: softTemplateText(row.positions),
      score,
      slots: row.slots.map((slot) => [...slot.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6)),
      positions: row.positions.map((position) => [...position.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5)),
      modelPositions: row.positions.map((position) => [...position.entries()].sort((a, b) => b[1] - a[1])),
      sectionCount: row.sections.size,
    };
  }).filter((row) => row.count >= 4 && row.slotVariety > row.slots.length)
    .sort((a, b) => b.score - a.score || b.count - a.count || b.slotVariety - a.slotVariety || b.template.length - a.template.length);

  const selected = [];
  const selectedFootprints = [];
  for (const row of ranked) {
    if (selectedFootprints.some((existing) => setOverlap(row.footprint, existing) >= 0.72)) continue;
    selected.push(row);
    selectedFootprints.push(row.footprint);
    if (selected.length >= 40) break;
  }
  return selected.map((row) => {
    const { occurrences, footprint, sections, ...rest } = row;
    return rest;
  });
}

function gappedTemplates(fragments, report) {
  const wildcardCache = new Map();
  for (let length = 5; length <= 9; length += 1) wildcardCache.set(length, wildcardSets(length));
  const rows = new Map();
  scanTemplatesForFragments(fragments, rows, wildcardCache, null, report);
  return rankAndDedupTemplates(rows);
}

function setOverlap(a, b) {
  let shared = 0;
  for (const item of a) if (b.has(item)) shared += 1;
  return shared / Math.max(1, Math.min(a.size, b.size));
}

function softTemplateText(positions) {
  return positions.map((position) => {
    const entries = [...position.entries()].sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((sum, [, count]) => sum + count, 0);
    if (!total) return "_";
    const [topToken, topCount] = entries[0];
    if (topCount / total >= 0.84) return topToken;
    return `{${entries.slice(0, 3).map(([token]) => token).join("/")}}`;
  }).join(" ");
}

function patternRows(fragments, report) {
  let processedWords = 0;
  return fragments.map((frag, position) => {
    processedWords += frag.tokens.length;
    const counts = new Map();
    const signatures = [];
    let repeatedWindows = 0;
    for (const n of [2, 3, 4]) {
      for (const phrase of ngrams(frag.tokens, n).map((window) => window.join(" "))) {
        counts.set(phrase, (counts.get(phrase) ?? 0) + 1);
      }
    }
    for (const [phrase, count] of counts.entries()) {
      if (count > 1) {
        repeatedWindows += count - 1;
        signatures.push(phrase);
      }
    }
    const tokenCounts = new Map();
    for (const token of frag.tokens) tokenCounts.set(token, (tokenCounts.get(token) ?? 0) + 1);
    const repeatedMass = [...tokenCounts.values()].reduce((sum, count) => sum + Math.max(0, count - 1), 0);
    const dominantShare = Math.max(0, ...tokenCounts.values()) / Math.max(1, frag.tokens.length);
    const uniqueRatio = tokenCounts.size / Math.max(1, frag.tokens.length);
    const repeatDensity = repeatedWindows / Math.max(1, frag.tokens.length - 1);
    const reasons = [];
    if (signatures.length) reasons.push("internal-window-repeat");
    if (repeatedMass >= 2) reasons.push("token-reuse");
    if (repeatDensity >= 0.12) reasons.push("looped-phrasing");
    if (uniqueRatio <= 0.72 && frag.tokens.length >= 6) reasons.push("low-diversity");
    if (dominantShare >= 0.22 && frag.tokens.length >= 8) reasons.push("dominant-token");
    const score = repeatedMass * 0.7 + repeatDensity * 2.3 + (1 - uniqueRatio) * 1.3 + dominantShare * 0.8;
    report?.({
      stage: "Checking inside-fragment patterns",
      stagePct: 76 + (position / Math.max(1, fragments.length)) * 6,
      metrics: { processedFragments: position + 1, processedWords, insideFlagsChecked: position + 1 },
    });
    return { fragment: frag, reasons, signatures, repeatedMass, dominantShare, uniqueRatio, repeatDensity, score };
  }).filter((row) => row.reasons.length).sort((a, b) => b.score - a.score || b.repeatDensity - a.repeatDensity);
}

function removalSpan(text, frag) {
  let left = frag.sourceStart;
  let right = frag.sourceEnd;
  while (left > 0 && /[ \t]/.test(text[left - 1])) left -= 1;
  while (right < text.length && /[ \t]/.test(text[right])) right += 1;
  const separators = ",.;:!?—–";
  let hasLeft = left > 0 && separators.includes(text[left - 1]);
  const hasRight = right < text.length && separators.includes(text[right]);
  if (hasLeft && text[left - 1] === ":" && isSpeakerColon(text, left - 1)) hasLeft = false;
  if (hasLeft && hasRight) left -= 1;
  else if (hasRight && !hasLeft) right += 1;
  else if (hasLeft) left -= 1;
  while (left > 0 && /[ \t]/.test(text[left - 1])) left -= 1;
  while (right < text.length && /[ \t]/.test(text[right])) right += 1;
  while (left > 0 && /["'“”‘’]/.test(text[left - 1])) left -= 1;
  while (right < text.length && /["'“”‘’]/.test(text[right])) right += 1;
  const dqRemoved = text.slice(left, right);
  const dqCount = (dqRemoved.match(/["\u201c\u201d]/g) || []).length;
  if (dqCount === 1) {
    const dq = /["\u201c\u201d]/;
    if (left < frag.sourceStart && dq.test(text[left])) {
      left += 1;
      while (left < frag.sourceStart && /[ \t]/.test(text[left])) left += 1;
    } else if (right > frag.sourceEnd && dq.test(text[right - 1])) {
      right -= 1;
      while (right > frag.sourceEnd && /[ \t]/.test(text[right - 1])) right -= 1;
    }
  }
  return { left, right };
}

function cleanup(text) {
  return text
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/\n{3,}(#{1,6}\s+)/g, "\n\n$1")
    .replace(/([^\n])\n(#{1,6}\s+)/g, "$1\n\n$2")
    .replace(/(#{1,6}[^\n]+)\n([^\n])/g, "$1\n\n$2");
}

function normalizeEditMarkers(text) {
  const gap = `[ \\t,.;!?—–"'“”‘’]*`;
  const cluster = new RegExp(`@(?:${gap}@)+`, "g");
  return text
    .replace(cluster, "@")
    .replace(/[ \t]*[,.;!?—–"'“”‘’]+[ \t]*@[ \t]*/g, " @ ")
    .replace(/[ \t]*@[ \t]*[,.;!?—–"'“”‘’]+[ \t]*/g, " @ ")
    .replace(cluster, "@");
}

function selectedRows(rows, fragmentCount, profile) {
  const maxRemove = Math.max(1, Math.floor(fragmentCount * profile.maxRemoveRatio));
  const relativeCutoff = rows.length >= maxRemove ? rows[maxRemove - 1].badness : 0;
  const selected = [];
  for (const row of rows) {
    if (selected.length >= maxRemove) break;
    if (row.hint !== "redundant-candidate") continue;
    if (row.fragment.startsCapitalized && !profile.allowCapitalized) continue;
    const clearsAbsolute = row.badness >= profile.minBadness;
    const clearsRelative = row.badness >= relativeCutoff && row.badness >= profile.minViable;
    if (!clearsAbsolute && !clearsRelative) continue;
    if (row.matchCount < profile.minMatches) continue;
    if (row.reusableRatio < profile.minReuse) continue;
    if (row.unique > profile.maxUnique) continue;
    selected.push(row);
  }
  return selected;
}

function patternModelFromTemplate(row) {
  if (!row) return null;
  return {
    length: row.modelPositions.length,
    positions: row.modelPositions.map((entries) => new Map(entries)),
    softText: row.softText,
    sourceCount: row.count,
  };
}

function scoreWindowAgainstModel(tokens, model) {
  let score = 0;
  let matched = 0;
  let peak = 0;
  for (let index = 0; index < model.length; index += 1) {
    const counter = model.positions[index];
    const total = [...counter.values()].reduce((sum, count) => sum + count, 0);
    const probability = total ? (counter.get(tokens[index]) ?? 0) / total : 0;
    score += probability;
    if (probability > 0) matched += 1;
    peak = Math.max(peak, probability);
  }
  return { score: score / Math.max(1, model.length), matched, peak };
}

function patternCandidates(fragments, model, minScore, report) {
  if (!model) return [];
  const out = [];
  const minMatched = Math.max(3, Math.floor(model.length * 0.68));
  let processedWords = 0;
  for (const [position, frag] of fragments.entries()) {
    processedWords += frag.tokens.length;
    if (frag.tokens.length < model.length) {
      report?.({
        stage: "Matching scaffold pass",
        stagePct: 82 + (position / Math.max(1, fragments.length)) * 12,
        metrics: { processedFragments: position + 1, processedWords, patternCandidates: out.length },
      });
      continue;
    }
    let best = null;
    for (let start = 0; start <= frag.tokens.length - model.length; start += 1) {
      const window = frag.tokens.slice(start, start + model.length);
      const scored = scoreWindowAgainstModel(window, model);
      if (scored.matched < minMatched || scored.score < minScore) continue;
      const candidate = { fragment: frag, window, start, ...scored };
      if (!best || candidate.score > best.score || (candidate.score === best.score && candidate.matched > best.matched)) best = candidate;
    }
    if (best) out.push(best);
    report?.({
      stage: "Matching scaffold pass",
      stagePct: 82 + (position / Math.max(1, fragments.length)) * 12,
      metrics: { processedFragments: position + 1, processedWords, patternCandidates: out.length },
    });
  }
  return out.sort((a, b) => a.fragment.paraIndex - b.fragment.paraIndex || a.fragment.idx - b.fragment.idx);
}

function selectPatternCandidates(candidates, policy) {
  const byPart = new Map();
  for (const candidate of candidates) {
    const key = candidate.fragment.paraIndex;
    if (!byPart.has(key)) byPart.set(key, []);
    byPart.get(key).push(candidate);
  }
  const selected = [];
  if (policy.startsWith("keep-")) {
    const keep = Number(policy.replace("keep-", ""));
    for (const group of byPart.values()) {
      // Sort by score descending: highest-scoring (most typical) first.
      // Remove the most typical instances, keep the atypical ones.
      group.sort((a, b) => b.score - a.score);
      selected.push(...group.slice(keep));
    }
    return selected;
  }
  if (policy === "threshold-dedup") {
    const scores = candidates.map((candidate) => candidate.score).sort((a, b) => a - b);
    const threshold = scores.length ? Math.max(0.58, scores[Math.floor(scores.length * 0.72)]) : 1;
    for (const group of byPart.values()) {
      const highConfidence = group.filter((candidate) => candidate.score >= threshold).sort((a, b) => b.score - a.score);
      selected.push(...highConfidence.slice(1));
    }
  }
  return selected;
}

function normalizeScores(items, getScore) {
  if (!items.length) return new Map();
  const scores = items.map(getScore);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const out = new Map();
  items.forEach((item, index) => {
    const raw = max === min ? 1 : (scores[index] - min) / (max - min);
    out.set(item, 0.22 + raw * 0.78);
  });
  return out;
}

function applySpans(text, spans) {
  let refloored = text;
  for (const span of spans) {
    refloored = refloored.slice(0, span.left) + " @ " + refloored.slice(span.right);
  }
  return cleanup(normalizeEditMarkers(refloored));
}

function mergedSpans(spans) {
  const sorted = spans.slice().sort((a, b) => a.left - b.left || a.right - b.right);
  const merged = [];
  for (const span of sorted) {
    const previous = merged[merged.length - 1];
    if (!previous || span.left > previous.right) {
      merged.push({ ...span });
      continue;
    }
    previous.right = Math.max(previous.right, span.right);
    if ((span.severity ?? 0) > (previous.severity ?? 0)) {
      previous.severity = span.severity;
      previous.method = span.method;
      previous.row = span.row;
    }
  }
  return merged;
}

export function wordCount(text) {
  return (text.match(/\b[\w'-]+\b/g) ?? []).length;
}

export function prepareDocumentForBatch(text, settings = DEFAULT_SETTINGS) {
  const { frontmatter, paragraphs } = parseParagraphs(text);
  const excluded = new Set([...extractCharacterTokens(frontmatter), ...extractSpeakerTokens(text)]);
  const fragments = paragraphs.flatMap((paragraph) => splitFragments(paragraph, excluded, settings.minTokens));
  return {
    paragraphs,
    fragments,
    originalWords: wordCount(text),
  };
}

export function buildBatchPatternModel(preparedDocs, report = null) {
  const wildcardCache = new Map();
  for (let length = 5; length <= 9; length += 1) wildcardCache.set(length, wildcardSets(length));
  const rows = new Map();
  let hotTokens = null;
  let globalIdx = 0;

  for (const [docIndex, doc] of preparedDocs.entries()) {
    const docFragments = doc.fragments.map((f) => ({ ...f, idx: globalIdx++, batchDocIndex: docIndex }));
    const isFiltered = hotTokens && hotTokens.size > 0;

    report?.({
      stage: `Scaffold scan: file ${docIndex + 1}/${preparedDocs.length}${isFiltered ? ` (${hotTokens.size} hot tokens)` : " (building hot token set)"}`,
      progressPct: 5 + (docIndex / preparedDocs.length) * 40,
      metrics: { templateSeeds: rows.size },
      force: true,
    });

    scanTemplatesForFragments(docFragments, rows, wildcardCache, hotTokens, (progress) => {
      const fragProgress = progress.metrics?.processedFragments ?? 0;
      report?.({
        ...progress,
        stage: `File ${docIndex + 1}/${preparedDocs.length}: ${progress.stage}${isFiltered ? ` (${hotTokens.size} hot tokens)` : ""}`,
        progressPct: 5 + ((docIndex + fragProgress / Math.max(1, docFragments.length)) / preparedDocs.length) * 40,
      });
    });

    hotTokens = extractHotTokens(rows);
  }

  report?.({
    stage: "Ranking scaffold templates",
    progressPct: 48,
    force: true,
  });

  const templateRows = rankAndDedupTemplates(rows);

  report?.({
    stage: "Scaffold prior built",
    progressPct: 50,
    force: true,
  });

  return {
    templateRows,
    patternModels: templateRows.map((row) => patternModelFromTemplate(row)).filter(Boolean),
  };
}

export function analyze(text, settings = DEFAULT_SETTINGS, profileName = "med", onProgress = null, options = {}) {
  const now = () => (globalThis.performance?.now ? globalThis.performance.now() : Date.now());
  const started = now();
  const timings = {};
  const originalWords = wordCount(text);
  const progressMetrics = {
    originalWords,
    processedWords: 0,
    processedFragments: 0,
    nearMatches: 0,
    stagedRemovals: 0,
    patternCandidates: 0,
    templateSeeds: 0,
  };
  const report = ({ stage, stagePct, metrics = {}, force = false }) => {
    Object.assign(progressMetrics, metrics);
    onProgress?.({
      stage,
      progressPct: Math.max(0, Math.min(99, stagePct)),
      metrics: { ...progressMetrics },
      force,
    });
  };

  const parseStart = now();
  report({ stage: "Parsing markdown", stagePct: 2, force: true });
  const { frontmatter, paragraphs } = parseParagraphs(text);
  const excluded = new Set([...extractCharacterTokens(frontmatter), ...extractSpeakerTokens(text)]);
  timings.parse = now() - parseStart;
  report({ stage: "Parsed markdown", stagePct: 8, metrics: { paragraphs: paragraphs.length }, force: true });

  const fragmentStart = now();
  const fragments = [];
  let indexedWords = 0;
  for (const [index, paragraph] of paragraphs.entries()) {
    const paragraphFragments = splitFragments(paragraph, excluded, settings.minTokens);
    fragments.push(...paragraphFragments);
    indexedWords += paragraphFragments.reduce((sum, fragment) => sum + fragment.tokens.length, 0);
    report({
      stage: "Indexing fragments",
      stagePct: 8 + (index / Math.max(1, paragraphs.length)) * 17,
      metrics: { processedWords: indexedWords, processedFragments: fragments.length },
    });
  }
  timings.fragment = now() - fragmentStart;
  report({
    stage: "Indexed fragments",
    stagePct: 25,
    metrics: { processedWords: indexedWords, processedFragments: fragments.length },
    force: true,
  });

  const profile = PROFILES[profileName];
  const effectiveSettings = {
    ...settings,
    frontierGrowth: settings.frontierGrowth > 0 ? settings.frontierGrowth : (profile.frontierGrowth ?? 1),
  };

  const scoreStart = now();
  const rows = scoreFragments(fragments, effectiveSettings, report);
  timings.score = now() - scoreStart;
  report({ stage: "Scored local repetition", stagePct: 55, metrics: { nearMatches: rows.length }, force: true });

  const selectStart = now();
  const profileSelected = selectedRows(rows, fragments.length, profile);
  const localSeverity = normalizeScores(profileSelected, (row) => row.badness);
  const profileSpans = profileSelected.map((row) => ({
    ...removalSpan(text, row.fragment),
    row,
    method: "local",
    severity: localSeverity.get(row) ?? 0.22,
  })).sort((a, b) => b.left - a.left);
  timings.select = now() - selectStart;
  report({ stage: "Selected local removals", stagePct: 58, metrics: { stagedRemovals: profileSelected.length }, force: true });

  const diagnosticStart = now();
  const templateRows = gappedTemplates(fragments, report);
  const phraseRows = repeatedPhrases(rows);
  const patternRowsResult = patternRows(fragments, report);
  timings.diagnostics = now() - diagnosticStart;
  report({
    stage: "Built diagnostics",
    stagePct: 82,
    metrics: { templateSeeds: templateRows.length, insideFlags: patternRowsResult.length },
    force: true,
  });

  const patternStart = now();
  const patternModels = options.patternModels ?? (templateRows.length ? templateRows.map((row) => patternModelFromTemplate(row)).filter(Boolean) : []);
  // Sort by sourceCount descending: peakiest patterns first for cascading removal
  patternModels.sort((a, b) => (b?.sourceCount ?? 0) - (a?.sourceCount ?? 0));

  const policy = settings.outputMode?.startsWith("pattern-") ? settings.outputMode.replace("pattern-", "") : "off";
  const patternSelected = [];
  if (policy !== "off" && patternModels.length) {
    const seenFragments = new Set();
    for (const model of patternModels) {
      if (!model) continue;
      const candidates = patternCandidates(fragments, model, settings.patternMinScore, report);
      const selected = selectPatternCandidates(candidates, policy);
      for (const cand of selected) {
        if (!seenFragments.has(cand.fragment.idx)) {
          seenFragments.add(cand.fragment.idx);
          patternSelected.push(cand);
        }
      }
    }
  }
  const patternCandidateRows = patternModels.length ? patternCandidates(fragments, patternModels[0], settings.patternMinScore, report) : [];
  const scaffoldSeverity = normalizeScores(patternSelected, (candidate) => candidate.score);
  const patternSpans = patternSelected.map((candidate) => ({
    ...removalSpan(text, candidate.fragment),
    row: { fragment: candidate.fragment, candidate },
    method: "scaffold",
    severity: scaffoldSeverity.get(candidate) ?? 0.22,
  })).sort((a, b) => b.left - a.left);
  timings.pattern = now() - patternStart;
  report({
    stage: "Selected scaffold removals",
    stagePct: 94,
    metrics: { patternCandidates: patternCandidateRows.length, stagedRemovals: profileSelected.length + patternSelected.length },
    force: true,
  });

  const selected = [...profileSelected, ...patternSelected.map((candidate) => ({ fragment: candidate.fragment, candidate }))];
  const spans = mergedSpans([...profileSpans, ...patternSpans]).sort((a, b) => b.left - a.left);

  const applyStart = now();
  report({ stage: "Applying removals", stagePct: 96, metrics: { stagedRemovals: spans.length }, force: true });
  const refloored = applySpans(text, spans);
  timings.apply = now() - applyStart;

  const keptWords = wordCount(refloored);
  timings.total = now() - started;
  report({
    stage: "Complete",
    stagePct: 99,
    metrics: {
      keptWords,
      strippedWords: Math.max(0, originalWords - keptWords),
      stagedRemovals: spans.length,
      patternCandidates: patternCandidateRows.length,
    },
    force: true,
  });

  return {
    paragraphs,
    fragments,
    rows,
    selected,
    spans: spans.slice().sort((a, b) => a.left - b.left),
    refloored,
    templateRows,
    phraseRows,
    patternRows: patternRowsResult,
    patternModel: patternModels[0] ?? null,
    patternCandidates: patternCandidateRows,
    outputMode: settings.outputMode,
    timings,
    metrics: {
      originalWords,
      keptWords,
      strippedWords: Math.max(0, originalWords - keptWords),
      keptPct: keptWords / Math.max(1, originalWords) * 100,
      fragments: fragments.length,
      nearMatchedFragments: rows.length,
      removedFragments: spans.length,
    },
  };
}
