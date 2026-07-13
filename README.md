# Ablation Lab

Ablation Lab finds and removes repetitive text patterns in AI-generated prose — the kind of patterns that emerge when an LLM's `repetitionPenalty` has been weakened (whether through aggressive tuning for speed, increased concurrency, or cost reduction) and the model starts cycling near its rejection boundary.

## Why

Many AI-generated documents suffer from near-duplicate sentences, scaffold phrasing loops ("X was the thing that Y"), and internal fragment repetition that a human editor would catch but automated thesaurus-based deduplication makes worse. These problems are especially visible in longer works — books, articles, corpus-scale content — where a weakened repetition penalty lets the model produce text that reads fine sentence-by-sentence but repeats heavily across the whole document.

Ablation Lab works better on larger corpuses because repetition patterns that are invisible at the paragraph level become obvious across hundreds of fragments. The tool grows its search window adaptively — the more repetition it finds around a fragment, the further it reaches to catch distant occurrences of the same pattern.

## What it does

- **Detects near-duplicate fragments** by tokenising thought-units (split on clause punctuation) and scoring them by token overlap, sequence similarity, and proximity
- **Excises redundant fragments** with responsible punctuation healing — removes separator punctuation, absorbs surrounding quotes, preserves newlines and heading structure
- **Flags inside-fragment patternisation** — the "X was the thing that Y" loops that can't be auto-excised but need rewrite attention
- **Detects diffuse scaffold templates** — repeated structural patterns with variable slots across the document
- **Optionally ablates scaffold occurrences** using a probability model built from the document's own patterns

## How to use

Upload a markdown file at [bobdavies.co.uk/ablation-lab/](https://bobdavies.co.uk/ablation-lab/). Choose a local pass strength (low/med/high) and a scaffold ablation policy. The tool runs entirely in your browser — no file is uploaded anywhere, no telemetry, no logging. Review the inline diff showing exactly what was removed and why, then export the cleaned markdown.

Multiple files can be dropped at once for batch processing with a shared corpus-wide pattern model.

## Privacy

Everything is client-side. Your documents never leave your machine. The only network request is the initial page load.

## License

DBAD (Don't Be A Dick).
