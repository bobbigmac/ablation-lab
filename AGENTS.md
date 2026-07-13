# AGENTS.md — Ablation Lab Technical Documentation

## Project Overview

Ablation Lab is a fully client-side tool for detecting and excising repetitive text patterns in AI-generated prose. It targets content produced by LLMs where the model's `repetitionPenalty` was insufficient or weakened, causing it to cycle near its rejection boundary — producing near-duplicate sentences, scaffold phrasing loops, and internal fragment patternisation that a human editor would catch but automated thesaurus-based deduplication worsens.

The tool operates on markdown documents (books, articles, any prose) and uses a token-nearness approach: it splits text into thought-unit fragments, tokenises them, builds an inverted index, and scores fragments by how many nearby fragments share meaningful token overlap. Fragments identified as redundant are excised with responsible punctuation healing, and diffuse scaffold patterns (repeated structural templates with variable slots) are detected and optionally ablated.

## Architecture

### Two implementations

The project contains two parallel implementations of the same algorithm:

1. **Python script** (`scripts/find-repetition-hotspots.py`) — CLI tool for batch processing, corpus-wide analysis, and report generation. Uses NLTK's `TreebankWordTokenizer` for tokenisation and `difflib.SequenceMatcher` for sequence similarity.

2. **JavaScript web app** (`src/`) — Vite + React SPA running entirely in the browser via a Web Worker. Accepts file uploads, processes client-side, shows inline diffs, and exports cleaned markdown. No server, no logging, no telemetry.

Both implementations share identical algorithm logic, profile parameters, and scoring formulas. The Python version is the original; the JS version is the interactive refinement.

### JavaScript application structure

- `src/refloor.js` — Core algorithm: parsing, tokenising, fragment scoring, pattern detection, span removal, punctuation healing. Exports `analyze()`, `prepareDocumentForBatch()`, `buildBatchPatternModel()`, `PROFILES`, `DEFAULT_SETTINGS`.
- `src/refloor.worker.js` — Web Worker wrapper that calls `analyze()` off the main thread. Handles both single-file and batch modes.
- `src/main.jsx` — React UI: file upload (click + drag-and-drop), parameter controls, inline diff rendering, diagnostics panel, batch workspace with per-file metrics and auto-download, in-app help modal documenting protected names, @ markers, profiles, and controls.
- `src/batch.js` — Batch utilities: SHA-256 hashing, job creation, localStorage history (metadata only, 250-row limit), ablation frontmatter prefixing, download trigger.
- `src/styles.css` — Full styling, dark sidebar + light workspace, severity-coloured diff highlights.
- `vite.config.js` — Vite config with `@vitejs/plugin-react` and configurable `base` path (defaults to `/ablation-lab/`).

## Algorithm

### 1. Document parsing

Documents are split into paragraphs by blank lines. Markdown headings (`#` through `######`) are treated as section boundaries, not content — they are excluded from fragment tokenisation and do not contribute to the repetition index. This ensures headings remain invisible to the detection system while still serving as paragraph delimiters. Frontmatter (YAML between `---` fences) is stripped and parsed separately for character name extraction.

### 2. Fragment splitting

Each paragraph's text is split into thought-unit fragments by clause/sentence punctuation: `, . ; : ! ? — –`. Apostrophes are preserved (no splitting on contractions or possessives). Each fragment is cleaned by stripping surrounding quotes, brackets, emphasis markers, and whitespace. Speaker labels (lines matching `Name:` at line start) are detected and excluded from fragmentation.

Fragments with fewer than `minTokens` tokens (default 3) are discarded — they lack enough signal for meaningful comparison.

### 3. Tokenisation

Tokens are extracted via regex (JS) or `TreebankWordTokenizer` (Python), normalised to lowercase with stripped non-alphanumeric edges. Hyphenated and apostrophe-bearing tokens are preserved as single units (e.g., `well-known`, `didn't`).

**All tokens participate in repetition detection.** There is no stopword filtering of content — mundane words like "the", "was", "thing" are intentionally included because the tool targets overuse of exactly these scaffold words. The user explicitly rejected stopword-based filtering of content tokens as a "filthy hack to avoid doing what was actually described."

### 4. Character name exclusion

Character names are excluded from the token index so that repeated proper names don't inflate repetition scores. Two extraction methods:

- **Frontmatter `characters:` field** — reads a single-line dense YAML entry (e.g., `characters: [Alice, Bob, Dr. Chen]`), tokenises each name, and adds non-stopword tokens to the exclusion set.
- **Speaker labels** — scans the document body for lines starting with `Name:` (dialogue attribution), tokenises the label, and adds to the exclusion set.

This is per-document, not global, because different documents have different character sets.

### 5. Inverted index and candidate finding

An inverted index maps each token to the list of fragment indices containing it. For each fragment, candidate comparison fragments are found by:

1. Looking up each of the fragment's tokens in the index
2. Filtering to fragments within `effectiveRadius` paragraphs (see adaptive growth below)
3. Skipping tokens with posting lists larger than `maxCandidatesPerToken` (default 180) — these are too common to be useful signals

This is the "fast" first pass: only fragments sharing at least one token are considered, avoiding O(n²) all-pairs comparison.

### 6. Adaptive frontier expansion

The core innovation. For each fragment, the system doesn't just compare against initial candidates — it grows a search frontier:

1. Start with `frontierBoost = 1` and the fragment's own tokens
2. Collect initial candidates within `paragraphRadius * min(1, growthMultiplier)` paragraphs
3. Score all candidates for similarity
4. Extract "bridge tokens" — tokens from the top-scoring matches that aren't already in the frontier
5. Use bridge tokens to find new candidates, now within `paragraphRadius * min(frontierBoost, growthMultiplier)` paragraphs
6. **Double `frontierBoost`** at each iteration
7. Repeat until no new bridge tokens are found or no new candidates emerge

This means patterns that repeat heavily expand their search window exponentially. A fragment with many near-matches will reach further across the document, catching occurrences that are quite far away. The `frontierGrowth` parameter caps how far the spatial window grows:

- **low profile: growthMultiplier = 1** — radius stays fixed at `paragraphRadius` (no spatial growth, only token frontier grows)
- **med profile: growthMultiplier = 3** — radius can grow up to 3× the base
- **high profile: growthMultiplier = 4** — radius can grow up to 4× the base

The UI "Growth" control allows overriding: 0 = auto (use profile default), 1 = fixed radius, higher = stronger growth.

### 7. Fragment similarity scoring

Each candidate pair is scored on multiple dimensions:

- **Jaccard overlap** (weight 0.38) — set intersection over union of token sets
- **Sequence similarity** (weight 0.30) — longest common subsequence ratio (Python: `difflib.SequenceMatcher`, JS: dynamic programming LCS)
- **Bigram Jaccard** (weight 0.16) — overlap of adjacent token pairs
- **Trigram Jaccard** (weight 0.08) — overlap of three-token windows
- **Size similarity** (weight 0.08) — penalty for length mismatch

An **exchange counterweight** detects I/You swaps: if one fragment contains "i" and the other "you" (or vice versa) with overlap ≥ 0.35, a 0.12 score reduction is applied. This prevents meaningful dialogue exchanges from being flagged as mere repetition.

### 8. Badness scoring

For each fragment with matches, a composite "badness" score is computed:

```
density = sum(match.weighted for each match)
  where weighted = adjustedScore / (1 + paragraphDistance)

badness = density * (1 + min(matchCount, 8) / 4) * (0.5 + reusableRatio)
  where reusableRatio = reusableTokens / totalUniqueTokens
```

More matches, closer proximity, and higher token reuse all increase badness. Fragments are ranked by badness descending.

### 9. Keep/Remove hints

For each cluster of similar fragments, a hint determines which to keep:

- **earliest** — the first occurrence in document order is always kept
- **earliest-capitalized** — if the earliest capitalized fragment differs from the earliest overall, it's also kept (capitalisation suggests an intentional sentence start)
- **redundant-candidate** — all others are candidates for removal

When all else is equal, later repetitions are preferentially removed (the user specified this preference).

### 10. Profile-based selection

Three profiles control how aggressively fragments are removed:

| Parameter | low | med | high |
|---|---|---|---|
| `minBadness` | 2.8 | 1.8 | 1.0 |
| `minViable` | 0.50 | 0.20 | 0.10 |
| `minMatches` | 2 | 1 | 1 |
| `minReuse` | 0.65 | 0.50 | 0.40 |
| `maxUnique` | 2 | 4 | 6 |
| `maxRemoveRatio` | 0.08 | 0.18 | 0.35 |
| `allowCapitalized` | false | true | true |
| `frontierGrowth` | 1 | 3 | 4 |

Selection uses both absolute thresholds (`minBadness`) and relative cutoffs (the badness value at the `maxRemoveRatio` percentile). A fragment must clear either the absolute or relative threshold, plus meet `minMatches`, `minReuse`, and `maxUnique` constraints. Capitalized fragments are protected in the low profile.

This dual absolute/relative approach handles documents with varying trash ratios — a 50% trash document and a 10% trash document both get sensible removal counts without a single percentage tune.

### 11. Punctuation healing

When a fragment is removed, the system heals surrounding punctuation responsibly:

1. Expand the removal span to include adjacent whitespace
2. Check for separator punctuation (`, . ; : ! ? — –`) on left and/or right sides
3. Remove the separator from one side: if both sides have separators, remove the left one; if only one side has one, remove that one
4. Speaker colons (`Name:`) are protected — the colon is not removed if it's a dialogue label
5. Absorb surrounding quotes (`" ' " " ' '`) into the removal span
6. **Newlines are never collapsed or stripped** — this prevents body paragraphs from being smashed into headings
7. After all spans are applied, a cleanup pass normalises multiple spaces, ensures headings have blank lines on both sides, and consolidates adjacent edit markers

This is integrated into the span calculation, not applied as a separate post-process. The user explicitly required that "removing it responsibly should be a fundamental part of how it keeps track of the process."

### 12. Inside-fragment patternisation detection

Beyond cross-fragment repetition, the system detects patterns *within* individual fragments — the "X was the thing that Y" style loops that can't be excised by removing the whole fragment but need to be flagged for rewrite attention.

Five pattern types are detected:

- **internal-window-repeat** — repeated n-grams (2, 3, 4) within the fragment
- **token-reuse** — 2+ tokens used more than once
- **looped-phrasing** — repeat density ≥ 0.12 (repeated windows per token)
- **low-diversity** — unique token ratio ≤ 0.72 for fragments with 6+ tokens
- **dominant-token** — single token accounts for ≥ 22% of fragment with 8+ tokens

Each flagged fragment gets a composite score. These are reported as diagnostics, not automatically removed.

### 13. Diffuse scaffold templates

The system detects repeated structural templates with variable slots — patterns like `[token] was the thing that [token]` where the fixed positions recur across many fragments but the wildcard slots vary.

For each fragment, sliding windows of 5–9 tokens are extracted. Each window is converted to a template by replacing 1–3 internal positions with wildcards. Templates are aggregated, counted, and scored by:

- Frequency (count of occurrences)
- Slot variety (number of distinct tokens in wildcard positions)
- Position variety (number of distinct tokens at each fixed position)
- Section spread (how many different sections the template appears in)
- Fixed-to-total ratio (templates with more fixed positions score higher)

Templates are deduplicated by footprint overlap (≥ 72% overlap means they're covering the same token positions). The top 40 are reported.

The highest-scoring template becomes a **pattern model** — a position-wise probability distribution that can be matched against new fragments to find scaffold occurrences for ablation.

### 14. Scaffold ablation pass

Using the pattern model, the system scans all fragments for windows matching the scaffold template. Each window is scored by position-wise probability. Candidates above `patternMinScore` (default 0.52) with sufficient matched positions (≥ 68% of template length) are selected for removal.

Three policies control removal:

- **ceiling (Light)** — only remove the strongest matches, keeping the first occurrence per paragraph
- **floor2 (Medium)** — keep up to 2 occurrences per paragraph, remove the rest
- **floor1 (Hard)** — keep only 1 occurrence per paragraph, remove the rest
- **off** — skip scaffold ablation entirely, only run local duplicate removal

### 15. Batch mode and corpus-wide analysis

The web app supports batch processing of multiple files. In batch mode:

1. All documents are parsed and fragmentised
2. A **shared pattern model** is built from the combined fragment pool — this detects scaffold patterns that recur across the entire corpus, not just within a single document
3. Each document is then processed individually using the shared model, so corpus-wide patterns are caught even in documents where they appear only once or twice

Batch results include per-file metrics (words kept/stripped, local vs scaffold removal counts), SHA-256 hashes for deduplication, and automatic download of cleaned outputs with ablation metadata prepended as frontmatter.

## Settings reference

| Setting | Default | Description |
|---|---|---|
| `paragraphRadius` | 12 (JS) / 6 (Python) | Base search radius in paragraphs for nearby fragment comparison |
| `frontierGrowth` | 0 (auto) | Cap for spatial growth multiplier during frontier expansion. 0 = use profile default |
| `minMatchScore` | 0.20 (JS) / 0.24 (Python) | Minimum similarity score for a fragment pair to count as a near-match |
| `minTokens` | 3 | Minimum tokens a fragment must have to be considered |
| `maxCandidatesPerToken` | 180 | Maximum posting list size before a token is skipped as too common |
| `patternMinScore` | 0.52 | Minimum probability score for scaffold pattern matching |
| `outputMode` | `pattern-ceiling` | Scaffold ablation policy: off, ceiling, floor1, floor2 |
| `top` | 30 (JS) / 40 (Python) | Maximum items in diagnostic reports |

## Output format

Cleaned documents preserve the original markdown structure. Removed fragments are replaced with ` @ ` edit markers during processing, then normalised — adjacent markers collapse, orphaned punctuation is absorbed, and the final text is cleaned. The `@` markers are not present in the final output; they're an intermediate representation used during span application.

In batch mode, each output file is prefixed with ablation metadata frontmatter:

```yaml
---
ablation_lab: {
  "tool": "ablation-lab.v1",
  "mode": "generative-prior ablation",
  "file": "example.md",
  "input_sha256": "...",
  "processed_at": "2026-07-13T...",
  "local_pass": "med",
  "scaffold_pass": "light",
  "settings": {...},
  "words": {"original": 50000, "kept": 42000, "stripped": 8000},
  "removals": {"local": 120, "scaffold": 45, "total": 165}
}
---
```

## Performance

All processing is client-side. The Web Worker prevents UI thread blocking. Progress is reported per-phase with throttled updates (max 1/second unless forced). Per-phase timings are captured and displayed:

- **parse** — frontmatter stripping, paragraph splitting
- **fragment** — tokenisation, fragment building, indexing
- **score** — similarity scoring, frontier expansion, badness calculation
- **select** — profile-based fragment selection
- **diagnostics** — scaffold template detection, phrase counting, inside-fragment flagging
- **pattern** — scaffold model matching and candidate selection
- **apply** — span application, punctuation healing, cleanup

The scoring phase is typically the most expensive, scaling with fragment count × average matches per fragment. The `maxCandidatesPerToken` cap prevents pathological cases where a single common token generates an enormous candidate set.

## Deployment

The app is deployed to GitHub Pages via a workflow (`.github/workflows/deploy-pages.yml`) that builds with Vite and deploys the `dist/` directory. The Vite base path is configurable via `VITE_BASE_PATH` environment variable, defaulting to `/ablation-lab/`.

Live URL: `bobdavies.co.uk/ablation-lab/`

## In-app help

The UI includes a help modal (opened via the "Help & Documentation" button in the sidebar) that documents:

- **Protected names** — the `characters:` frontmatter field format, how tokens are extracted and excluded, and the speaker-label auto-extraction
- **@ markers** — how removed fragments are marked in output, consolidation behaviour, and how to use them as edit points
- **Profiles** — the three local pass profiles (low/med/high) and their thresholds
- **Scaffold Pass** — the four ablation levels (Off/Light/Medium/Hard)
- **Search controls** — what each parameter does (Radius, Match, Ablate, Min tokens, Posting cap, Growth)
- **Batch mode** — multi-file processing, auto-download, localStorage history
- **Privacy** — fully client-side, no server or telemetry

## License

DBAD (Don't Be A Dick).
