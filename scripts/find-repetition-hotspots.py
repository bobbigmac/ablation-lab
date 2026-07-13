#!/usr/bin/env python3
"""
Find local repetition hotspots in a markdown document.

This works at thought-unit granularity:
- split prose into comma/period-ish fragments
- tokenize fragments with TreebankWordTokenizer
- ignore repeated proper names so characters do not inflate repetition scores
- for each fragment, compare only nearby fragments that share tokens
- rank fragments by local near-clone density and reusable-token overlap

Usage:
    python3 scripts/find-repetition-hotspots.py --file path/to/document.md
    python3 scripts/find-repetition-hotspots.py --file document.md --profiles low,med,high --write-refloors output/
"""

from __future__ import annotations

import argparse
import difflib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nltk.tokenize import TreebankWordTokenizer

BOOKS_DIR = Path("content")
WORD_TOKENIZER = TreebankWordTokenizer()

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those",
    "with", "from", "by", "not", "no", "so", "just", "very", "also", "about", "what",
    "when", "where", "how", "why", "who", "which", "than", "then", "there", "their",
    "them", "into", "your", "our", "his", "her", "its", "over", "after", "before",
    "while", "because", "though", "although", "through", "during", "without", "within",
    "up", "down", "off", "again", "once", "only", "more", "most", "some", "any", "all",
    "each", "every", "few", "many", "much", "same", "other", "one", "two", "three",
    "as", "than", "then", "there", "here", "by", "from",
}


@dataclass(frozen=True)
class Paragraph:
    section: str
    para_index: int
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class Fragment:
    idx: int
    section: str
    para_index: int
    start_line: int
    end_line: int
    text: str
    source_text: str
    source_start: int
    source_end: int
    starts_capitalized: bool
    tokens: tuple[str, ...]


def strip_frontmatter(text: str) -> tuple[str, str]:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[: end + 5], text[end + 5 :]
    return "", text


def normalize_token(tok: str) -> str | None:
    tok = tok.lower()
    if not tok or not re.search(r"[a-z0-9]", tok):
        return None
    tok = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", tok)
    return tok or None


def extract_character_tokens(frontmatter: str) -> set[str]:
    tokens: set[str] = set()
    lines = frontmatter.splitlines()
    in_characters = False

    def add_phrase(phrase: str) -> None:
        for raw in WORD_TOKENIZER.tokenize(phrase):
            tok = normalize_token(raw)
            if tok and tok not in STOPWORDS:
                tokens.add(tok)

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if in_characters:
                break
            continue
        if line.startswith("characters:"):
            in_characters = True
            payload = line.split(":", 1)[1].strip()
            if payload.startswith("[") and payload.endswith("]"):
                payload = payload[1:-1]
                for item in payload.split(","):
                    add_phrase(item.strip().strip("\"'"))
                break
            if payload:
                for item in payload.split(","):
                    add_phrase(item.strip().strip("\"'"))
                continue
            continue
        if in_characters:
            if re.match(r"^[A-Za-z0-9_-]+:\s*", line) and not line.startswith("-"):
                break
            if line.startswith("-"):
                line = line[1:].strip()
            add_phrase(line.split("(", 1)[0].strip().rstrip(","))

    return tokens


def extract_speaker_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_line in text.splitlines():
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9 _.'’-]{0,48}):\s+", raw_line)
        if not match:
            continue
        for raw in WORD_TOKENIZER.tokenize(match.group(1).replace("\"", "").replace("'", "")):
            tok = normalize_token(raw)
            if tok and tok not in STOPWORDS:
                tokens.add(tok)
    return tokens


def tokenize_words(text: str, *, excluded_tokens: set[str] | None = None) -> tuple[str, ...]:
    out: list[str] = []
    excluded_tokens = excluded_tokens or set()
    for raw in WORD_TOKENIZER.tokenize(text):
        tok = normalize_token(raw)
        if tok and tok not in excluded_tokens:
            out.append(tok)
    return tuple(out)


def first_alpha(text: str) -> str | None:
    match = re.search(r"[A-Za-z]", text)
    return match.group(0) if match else None


def clean_fragment(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\s\"'“”‘’*_(){}\[\]<>-]+", "", text)
    text = re.sub(r"[\s\"'“”‘’*_(){}\[\]<>-]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_fragments(text: str) -> list[tuple[str, int, int]]:
    # Apostrophes are left alone; only clause/sentence punctuation creates thought units.
    spans: list[tuple[str, int, int]] = []
    matches = list(re.finditer(r"[^,.;:!?—–]+", text))
    for match in matches:
        raw = match.group(0)
        frag = clean_fragment(raw)
        if re.match(r"^[A-Za-z][A-Za-z0-9 _.'’-]{0,48}$", frag) and text[match.end() : match.end() + 1] == ":":
            continue
        if frag:
            spans.append((frag, match.start(), match.end()))
    return spans


def parse_paragraphs(text: str) -> list[Paragraph]:
    _, body = strip_frontmatter(text)
    lines = body.splitlines()

    paragraphs: list[Paragraph] = []
    current_section = "(preamble)"
    current_parts: list[str] = []
    current_start = 0
    para_index = 0

    def flush(end_line: int) -> None:
        nonlocal current_parts, current_start, para_index
        raw = "\n".join(current_parts).strip()
        if not raw:
            current_parts = []
            return
        paragraphs.append(
            Paragraph(
                section=current_section,
                para_index=para_index,
                start_line=current_start,
                end_line=end_line,
                text=raw,
            )
        )
        para_index += 1
        current_parts = []

    for lineno, raw_line in enumerate(lines, 1):
        stripped = raw_line.strip()
        if re.match(r"#{1,6}\s+", stripped):
            flush(lineno - 1)
            current_section = re.sub(r"^#{1,6}\s+", "", stripped).strip() or current_section
            continue
        if stripped == "---":
            flush(lineno - 1)
            continue
        if not stripped:
            flush(lineno - 1)
            continue
        if not current_parts:
            current_start = lineno
        current_parts.append(stripped)

    flush(len(lines))
    return paragraphs


def build_fragments(
    paragraphs: list[Paragraph], *, min_tokens: int, excluded_tokens: set[str]
) -> list[Fragment]:
    fragments: list[Fragment] = []
    for para in paragraphs:
        for raw_fragment, start, end in split_fragments(para.text):
            tokens = tokenize_words(raw_fragment, excluded_tokens=excluded_tokens)
            if len(tokens) < min_tokens:
                continue
            alpha = first_alpha(raw_fragment)
            fragments.append(
                Fragment(
                    idx=len(fragments),
                    section=para.section,
                    para_index=para.para_index,
                    start_line=para.start_line,
                    end_line=para.end_line,
                    text=raw_fragment,
                    source_text=para.text,
                    source_start=start,
                    source_end=end,
                    starts_capitalized=bool(alpha and alpha.isupper()),
                    tokens=tokens,
                )
            )
    return fragments


def ngrams(tokens: tuple[str, ...], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tokens[i : i + n] for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def seq_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def fragment_similarity(a: Fragment, b: Fragment) -> dict[str, float]:
    raw_overlap = jaccard(set(a.tokens), set(b.tokens))
    raw_seq = seq_similarity(a.tokens, b.tokens)
    bigram = jaccard(ngrams(a.tokens, 2), ngrams(b.tokens, 2))
    trigram = jaccard(ngrams(a.tokens, 3), ngrams(b.tokens, 3))
    size = 1.0 - abs(len(a.tokens) - len(b.tokens)) / max(len(a.tokens), len(b.tokens))

    shared_raw = len(set(a.tokens) & set(b.tokens))
    token_a = set(a.tokens)
    token_b = set(b.tokens)
    exchange_pair = (
        ("i" in token_a and "you" in token_b)
        or ("you" in token_a and "i" in token_b)
    ) and raw_overlap >= 0.35
    score = (
        0.38 * raw_overlap
        + 0.30 * raw_seq
        + 0.16 * bigram
        + 0.08 * trigram
        + 0.08 * size
    )
    return {
        "score": score,
        "raw_overlap": raw_overlap,
        "content_overlap": raw_overlap,
        "raw_seq": raw_seq,
        "content_seq": raw_seq,
        "bigram": bigram,
        "trigram": trigram,
        "size": size,
        "shared_content": shared_raw,
        "shared_raw": shared_raw,
        "exchange_pair": exchange_pair,
    }


def build_token_index(fragments: list[Fragment]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for frag in fragments:
        for tok in set(frag.tokens):
            index[tok].append(frag.idx)
    return index


def candidate_ids_for_tokens(
    frag: Fragment,
    fragments: list[Fragment],
    index: dict[str, list[int]],
    tokens: Iterable[str],
    *,
    effective_radius: int,
    max_candidates_per_token: int,
) -> set[int]:
    ids: set[int] = set()
    for tok in set(tokens):
        posting = index.get(tok, [])
        if len(posting) > max_candidates_per_token:
            continue
        for idx in posting:
            if idx == frag.idx:
                continue
            other = fragments[idx]
            if abs(other.para_index - frag.para_index) <= effective_radius:
                ids.add(idx)
    return ids


def nearby_candidates(
    frag: Fragment,
    fragments: list[Fragment],
    index: dict[str, list[int]],
    *,
    effective_radius: int,
    max_candidates_per_token: int,
) -> set[int]:
    return candidate_ids_for_tokens(
        frag,
        fragments,
        index,
        set(frag.tokens),
        effective_radius=effective_radius,
        max_candidates_per_token=max_candidates_per_token,
    )


def keep_hint(frag: Fragment, matches: list[dict], fragments: list[Fragment]) -> str:
    cluster = [frag] + [fragments[m["idx"]] for m in matches]
    earliest = min(cluster, key=lambda item: item.idx)
    capitalized = [item for item in cluster if item.starts_capitalized]
    earliest_cap = min(capitalized, key=lambda item: item.idx) if capitalized else None
    if frag.idx == earliest.idx:
        return "earliest"
    if earliest_cap and frag.idx == earliest_cap.idx:
        return "earliest-capitalized"
    if frag.starts_capitalized and earliest_cap and earliest_cap.idx == frag.idx:
        return "capitalized"
    return "redundant-candidate"


def score_fragments(
    fragments: list[Fragment],
    *,
    paragraph_radius: int,
    frontier_growth: int,
    min_match_score: float,
    min_shared_content: int,
    max_candidates_per_token: int,
    excluded_tokens: set[str],
) -> list[dict]:
    index = build_token_index(fragments)
    rows: list[dict] = []
    growth_multiplier = max(1, frontier_growth)

    for frag in fragments:
        effective_radius = paragraph_radius * min(1, growth_multiplier)
        candidate_ids = nearby_candidates(
            frag,
            fragments,
            index,
            effective_radius=effective_radius,
            max_candidates_per_token=max_candidates_per_token,
        )
        matches = []
        matched_ids: set[int] = set()
        frontier_tokens = set(frag.tokens)
        frontier_boost = 1

        def collect_matches(ids: set[int]) -> None:
            for idx in ids:
                if idx in matched_ids:
                    continue
                other = fragments[idx]
                metrics = fragment_similarity(frag, other)
                if metrics["shared_content"] < min_shared_content and metrics["bigram"] == 0:
                    continue
                if metrics["score"] < min_match_score:
                    continue
                adjusted_score = metrics["score"] - (0.12 if metrics["exchange_pair"] else 0.0)
                adjusted_score = max(0.0, adjusted_score)
                distance = abs(other.para_index - frag.para_index)
                proximity = 1.0 / (1 + distance)
                weighted = adjusted_score * proximity
                matches.append(
                    {
                        "idx": idx,
                        "metrics": metrics,
                        "distance": distance,
                        "weighted": weighted,
                    }
                )
                matched_ids.add(idx)

        collect_matches(candidate_ids)
        matches.sort(key=lambda m: (-m["weighted"], m["distance"], m["idx"]))

        while matches:
            bridge_tokens: list[str] = []
            seen_tokens = set(frontier_tokens) | excluded_tokens
            seed_count = min(len(matches), 4 * frontier_boost)
            for match in matches[:seed_count]:
                other = fragments[match["idx"]]
                for tok in other.tokens:
                    if tok in seen_tokens:
                        continue
                    seen_tokens.add(tok)
                    bridge_tokens.append(tok)
                if len(bridge_tokens) >= 4 * frontier_boost:
                    break

            if not bridge_tokens:
                break

            expand_radius = paragraph_radius * min(frontier_boost, growth_multiplier)
            expanded_ids = candidate_ids_for_tokens(
                frag,
                fragments,
                index,
                bridge_tokens,
                effective_radius=expand_radius,
                max_candidates_per_token=max_candidates_per_token,
            ) - matched_ids

            if not expanded_ids:
                break

            frontier_tokens.update(bridge_tokens)
            frontier_boost *= 2
            collect_matches(expanded_ids)
            matches.sort(key=lambda m: (-m["weighted"], m["distance"], m["idx"]))

        if not matches:
            continue

        local_token_counts: Counter[str] = Counter()
        for match in matches:
            local_token_counts.update(fragments[match["idx"]].tokens)

        reusable = sum(1 for tok in set(frag.tokens) if local_token_counts[tok] > 0)
        unique = len(set(frag.tokens)) - reusable
        reusable_ratio = reusable / max(1, len(set(frag.tokens)))
        density = sum(m["weighted"] for m in matches)
        badness = density * (1 + min(len(matches), 8) / 4) * (0.5 + reusable_ratio)

        rows.append(
            {
                "fragment": frag,
                "matches": matches,
                "match_count": len(matches),
                "density": density,
                "badness": badness,
                "reusable": reusable,
                "unique": unique,
                "reusable_ratio": reusable_ratio,
                "hint": keep_hint(frag, matches, fragments),
            }
        )

    rows.sort(
        key=lambda row: (
        -row["badness"],
        -row["match_count"],
        -row["reusable_ratio"],
        -row["fragment"].para_index,
        -row["fragment"].idx,
        )
    )
    return rows


def repeated_phrases(rows: list[dict], *, min_count: int = 4, max_items: int = 30) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        tokens = row["fragment"].tokens
        for n in (2, 3, 4):
            counts.update(" ".join(window) for window in ngrams(tokens, n))
    return [
        (phrase, count)
        for phrase, count in counts.most_common(max_items)
        if count >= min_count
    ]


def gapped_template_rows(
    fragments: list[Fragment],
    *,
    min_count: int = 4,
    max_items: int = 40,
) -> list[dict]:
    templates: dict[tuple[str, ...], dict] = {}

    def wildcard_sets(length: int) -> list[tuple[int, ...]]:
        internal = range(1, length)
        max_wildcards = min(3, max(1, length - 4))
        out: list[tuple[int, ...]] = []
        for wildcard_count in range(1, max_wildcards + 1):
            # Avoid itertools import churn in the hot path by using a tiny local recursive combiner.
            combo: list[int] = []

            def choose(start: int) -> None:
                if len(combo) == wildcard_count:
                    out.append(tuple(combo))
                    return
                for pos in range(start, length):
                    combo.append(pos)
                    choose(pos + 1)
                    combo.pop()

            choose(1)
        return out

    wildcard_cache = {length: wildcard_sets(length) for length in range(5, 10)}

    for frag in fragments:
        tokens = frag.tokens
        for length in range(5, 10):
            if len(tokens) < length:
                continue
            for start in range(0, len(tokens) - length + 1):
                window = tokens[start : start + length]
                for wildcard_positions in wildcard_cache[length]:
                    fixed = [tok if idx not in wildcard_positions else "_" for idx, tok in enumerate(window)]
                    if len(set(tok for tok in fixed if tok != "_")) < 3:
                        continue
                    template = tuple(fixed)
                    row = templates.setdefault(
                        template,
                        {
                            "template": template,
                            "count": 0,
                            "slots": [Counter() for _ in wildcard_positions],
                            "positions": [Counter() for _ in range(length)],
                            "occurrences": set(),
                            "footprint": set(),
                            "examples": [],
                            "sections": Counter(),
                        },
                    )
                    occurrence = (frag.idx, start)
                    if occurrence in row["occurrences"]:
                        continue
                    row["occurrences"].add(occurrence)
                    row["footprint"].update((frag.idx, start + pos) for pos in range(length))
                    row["count"] += 1
                    row["sections"][frag.section] += 1
                    for pos, token in enumerate(window):
                        row["positions"][pos][token] += 1
                    for slot_idx, pos in enumerate(wildcard_positions):
                        row["slots"][slot_idx][window[pos]] += 1
                    if len(row["examples"]) < 4:
                        example = " ".join(window)
                        if example not in row["examples"]:
                            row["examples"].append(example)

    rows: list[dict] = []
    for row in templates.values():
        if row["count"] < min_count:
            continue
        slot_variety = sum(len(slot) for slot in row["slots"])
        if slot_variety <= len(row["slots"]):
            continue
        fixed_count = sum(1 for tok in row["template"] if tok != "_")
        wildcard_count = len(row["slots"])
        section_count = len(row["sections"])
        row["text"] = " ".join(row["template"])
        row["slot_variety"] = slot_variety
        row["position_variety"] = sum(len(position) for position in row["positions"])
        row["soft_text"] = soft_template_text(row["positions"])
        row["score"] = (
            row["count"]
            * (1 + min(slot_variety, 24) / 8)
            * (1 + wildcard_count / 3)
            * (fixed_count / len(row["template"]))
            * (1 + min(section_count, 4) / 8)
        )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -row["score"],
            -row["count"],
            -row["slot_variety"],
            -len(row["template"]),
            row["text"],
        )
    )

    selected: list[dict] = []
    selected_footprints: list[set[tuple[int, int]]] = []
    for row in rows:
        footprint = row["footprint"]
        if any(len(footprint & existing) / max(1, min(len(footprint), len(existing))) >= 0.72 for existing in selected_footprints):
            continue
        selected.append(row)
        selected_footprints.append(footprint)
        if len(selected) >= max_items:
            break
    return selected


def soft_template_text(positions: list[Counter[str]]) -> str:
    parts: list[str] = []
    for counter in positions:
        total = sum(counter.values())
        if not total:
            parts.append("_")
            continue
        common = counter.most_common(3)
        top_token, top_count = common[0]
        if top_count / total >= 0.84:
            parts.append(top_token)
        else:
            parts.append("{" + "/".join(token for token, _ in common) + "}")
    return " ".join(parts)


def fragment_pattern_metrics(frag: Fragment) -> dict[str, object]:
    tokens = frag.tokens
    token_count = len(tokens)
    unique_tokens = len(set(tokens))

    repeated_windows = 0
    repeated_signatures: list[str] = []
    for n in (2, 3, 4):
        windows = [tokens[i : i + n] for i in range(max(0, token_count - n + 1))]
        counts = Counter(windows)
        for window, count in counts.items():
            if count > 1:
                repeated_windows += count - 1
                repeated_signatures.append(" ".join(window))

    counts = Counter(tokens)
    repeated_mass = sum(count - 1 for count in counts.values() if count > 1)
    dominant_share = max(counts.values()) / max(1, token_count)
    repeat_density = repeated_windows / max(1, token_count - 1)
    uniqueness_ratio = unique_tokens / max(1, token_count)

    reasons: list[str] = []
    if repeated_signatures:
        reasons.append("internal-window-repeat")
    if repeated_mass >= 2:
        reasons.append("token-reuse")
    if repeat_density >= 0.12:
        reasons.append("looped-phrasing")
    if uniqueness_ratio <= 0.72 and token_count >= 6:
        reasons.append("low-diversity")
    if dominant_share >= 0.22 and token_count >= 8:
        reasons.append("dominant-token")

    pattern_score = (
        repeated_mass * 0.7
        + repeat_density * 2.3
        + (1.0 - uniqueness_ratio) * 1.3
        + dominant_share * 0.8
        + (0.6 if "internal-window-repeat" in reasons else 0.0)
        + (0.5 if "token-reuse" in reasons else 0.0)
        + (0.4 if "low-diversity" in reasons else 0.0)
        + (0.3 if "dominant-token" in reasons else 0.0)
    )

    return {
        "pattern_score": pattern_score,
        "repeat_density": repeat_density,
        "repeated_mass": repeated_mass,
        "dominant_share": dominant_share,
        "uniqueness_ratio": uniqueness_ratio,
        "unique_tokens": unique_tokens,
        "repeated_signatures": repeated_signatures,
        "reasons": reasons,
    }


def flag_patternized_fragments(fragments: list[Fragment]) -> list[dict]:
    rows: list[dict] = []
    for frag in fragments:
        metrics = fragment_pattern_metrics(frag)
        if not metrics["reasons"]:
            continue
        rows.append({"fragment": frag, "metrics": metrics})
    rows.sort(
        key=lambda row: (
            -row["metrics"]["pattern_score"],
            -row["metrics"]["repeat_density"],
            -row["fragment"].para_index,
            -row["fragment"].idx,
        )
    )
    return rows


REFLOOR_PROFILES = {
    "low": {
        "min_badness": 2.8,
        "min_viable": 0.50,
        "min_matches": 2,
        "min_reuse": 0.65,
        "max_unique": 2,
        "max_remove_ratio": 0.08,
        "allow_capitalized": False,
        "frontier_growth": 1,
    },
    "med": {
        "min_badness": 1.8,
        "min_viable": 0.20,
        "min_matches": 1,
        "min_reuse": 0.50,
        "max_unique": 4,
        "max_remove_ratio": 0.18,
        "allow_capitalized": True,
        "frontier_growth": 3,
    },
    "high": {
        "min_badness": 1.0,
        "min_viable": 0.10,
        "min_matches": 1,
        "min_reuse": 0.40,
        "max_unique": 6,
        "max_remove_ratio": 0.35,
        "allow_capitalized": True,
        "frontier_growth": 4,
    },
}


def selected_for_profile(rows: list[dict], fragment_count: int, profile: dict) -> list[dict]:
    selected = []
    max_remove = max(1, int(fragment_count * profile["max_remove_ratio"]))
    relative_cutoff = rows[max_remove - 1]["badness"] if len(rows) >= max_remove else 0.0
    for row in rows:
        if len(selected) >= max_remove:
            break
        if row["hint"] != "redundant-candidate":
            continue
        if row["fragment"].starts_capitalized and not profile["allow_capitalized"]:
            continue
        clears_absolute = row["badness"] >= profile["min_badness"]
        clears_relative = row["badness"] >= relative_cutoff and row["badness"] >= profile["min_viable"]
        if not (clears_absolute or clears_relative):
            continue
        if row["match_count"] < profile["min_matches"]:
            continue
        if row["reusable_ratio"] < profile["min_reuse"]:
            continue
        if row["unique"] > profile["max_unique"]:
            continue
        selected.append(row)
    return selected


def cleanup_text(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}(#{1,6}\s+)", r"\n\n\1", text)
    text = re.sub(r"([^\n])\n(#{1,6}\s+)", r"\1\n\n\2", text)
    text = re.sub(r"(#{1,6}[^\n]+)\n([^\n])", r"\1\n\n\2", text)
    return text


def absorb_boundary_quotes(text: str, left: int, right: int) -> tuple[int, int]:
    quote_chars = "\"'“”‘’"
    while left > 0 and text[left - 1] in quote_chars:
        left -= 1
    while right < len(text) and text[right] in quote_chars:
        right += 1
    return left, right


def is_speaker_colon(text: str, colon_index: int) -> bool:
    line_start = text.rfind("\n", 0, colon_index) + 1
    label = text[line_start:colon_index].strip()
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9 _.'’-]{0,48}$", label))


def removal_span(text: str, frag: Fragment) -> tuple[int, int]:
    start = frag.source_start
    end = frag.source_end
    left = start
    right = end
    while left > 0 and text[left - 1] in " \t":
        left -= 1
    while right < len(text) and text[right] in " \t":
        right += 1

    separator_chars = ",.;:!?—–"
    has_left_sep = left > 0 and text[left - 1] in separator_chars
    has_right_sep = right < len(text) and text[right] in separator_chars
    if has_left_sep and text[left - 1] == ":" and is_speaker_colon(text, left - 1):
        has_left_sep = False

    if has_left_sep and has_right_sep:
        left -= 1
        while left > 0 and text[left - 1] in " \t":
            left -= 1
    elif has_right_sep and not has_left_sep:
        right += 1
        while right < len(text) and text[right] in " \t":
            right += 1
    elif has_left_sep:
        left -= 1
        while left > 0 and text[left - 1] in " \t":
            left -= 1
    else:
        while right < len(text) and text[right] in " \t":
            right += 1
    return absorb_boundary_quotes(text, left, right)


def normalize_edit_markers(text: str) -> str:
    marker_gap = r"[ \t,.;!?—–\"'“”‘’]*"
    text = re.sub(rf"@(?:{marker_gap}@)+", "@", text)
    text = re.sub(r"[ \t]*[,.;!?—–\"'“”‘’]+[ \t]*@[ \t]*", " @ ", text)
    text = re.sub(r"[ \t]*@[ \t]*[,.;!?—–\"'“”‘’]+[ \t]*", " @ ", text)
    text = re.sub(rf"@(?:{marker_gap}@)+", "@", text)
    return text


def apply_profile(text: str, selected: list[dict]) -> str:
    cleaned = text
    for row in sorted(selected, key=lambda row: (row["fragment"].para_index, row["fragment"].idx), reverse=True):
        frag = row["fragment"]
        match = re.search(re.escape(frag.text), cleaned)
        if not match:
            continue
        start, end = match.span()
        left, right = removal_span(cleaned, Fragment(
            idx=frag.idx,
            section=frag.section,
            para_index=frag.para_index,
            start_line=frag.start_line,
            end_line=frag.end_line,
            text=frag.text,
            source_text=cleaned,
            source_start=start,
            source_end=end,
            starts_capitalized=frag.starts_capitalized,
            tokens=frag.tokens,
        ))
        cleaned = cleaned[:left] + " @ " + cleaned[right:]
    return cleanup_text(normalize_edit_markers(cleaned))


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def write_refloors(
    text: str,
    path: Path,
    rows: list[dict],
    fragments: list[Fragment],
    out_dir: Path,
    profile_names: list[str],
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    original_words = word_count(text)
    for name in profile_names:
        profile = REFLOOR_PROFILES[name]
        selected = selected_for_profile(rows, len(fragments), profile)
        out_path = out_dir / f"{path.stem}.{name}.md"
        refloored = apply_profile(text, selected)
        kept_words = word_count(refloored)
        out_path.write_text(refloored, encoding="utf-8")
        written.append(
            {
                "profile": name,
                "path": out_path,
                "removed_fragments": len(selected),
                "original_words": original_words,
                "kept_words": kept_words,
                "stripped_words": max(0, original_words - kept_words),
                "kept_pct": kept_words / max(1, original_words) * 100,
            }
        )
    return written


def write_pattern_report(
    path: Path,
    rows: list[dict],
    fragments: list[Fragment],
    template_rows: list[dict],
    phrase_rows: list[tuple[str, int]],
    pattern_rows: list[dict],
    out_dir: Path,
    *,
    top: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.inside-fragments.md"
    lines: list[str] = []

    lines.append(f"# Repetition Hotspots: {path.name}")
    lines.append("")
    lines.append(f"- fragments: {len(fragments)}")
    lines.append(f"- fragments with local near-matches: {len(rows)}")
    lines.append("")
    lines.append("## Diffuse Scaffold Templates")
    lines.append("")
    if template_rows:
        for rank, row in enumerate(template_rows, 1):
            position_bits = []
            for idx, position in enumerate(row["positions"], 1):
                values = ", ".join(f"`{tok}` ({count})" for tok, count in position.most_common(5))
                position_bits.append(f"{idx}: {values}")
            examples = "; ".join(f"`{example}`" for example in row["examples"])
            lines.append(
                f"### {rank}. `{row['soft_text']}` "
                f"({row['count']} hits, {row['position_variety']} positional variants, {len(row['sections'])} sections)"
            )
            lines.append(f"- positions: {'; '.join(position_bits)}")
            if examples:
                lines.append(f"- examples: {examples}")
            lines.append("")
    else:
        lines.append("_No diffuse scaffold templates met the threshold._")
        lines.append("")
    lines.append("## Short Phrase Diagnostics")
    lines.append("")
    if phrase_rows:
        lines.append(", ".join(f"`{phrase}` ({count})" for phrase, count in phrase_rows))
    else:
        lines.append("_No repeated local phrase patterns met the threshold._")
    lines.append("")
    lines.append("## Diagnostic Pattern Summary")
    lines.append("")
    reason_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    for row in pattern_rows:
        reason_counts.update(row["metrics"]["reasons"])
        window_counts.update(row["metrics"]["repeated_signatures"])
    if reason_counts:
        lines.append(", ".join(f"`{reason}` ({count})" for reason, count in reason_counts.most_common(20)))
    else:
        lines.append("_No diagnostic pattern families were detected._")
    if window_counts:
        lines.append("")
        lines.append("Most repeated internal windows:")
        lines.append(", ".join(f"`{phrase}` ({count})" for phrase, count in window_counts.most_common(30)))
    lines.append("")
    lines.append("## Inside-Fragment Pattern Flags")
    lines.append("")
    if pattern_rows:
        for rank, row in enumerate(pattern_rows[:top], 1):
            frag = row["fragment"]
            metrics = row["metrics"]
            reasons = ", ".join(metrics["reasons"])
            lines.append(f"### {rank}. `{frag.section}` ({reasons})")
            lines.append(
                f"- shape: repeat={metrics['repeat_density']:.2f}, reuse={metrics['repeated_mass']}, "
                f"diversity={metrics['uniqueness_ratio']:.2f}, dominant={metrics['dominant_share']:.2f}"
            )
            lines.append(f"- fragment: {snippet(frag.text)}")
            if metrics["repeated_signatures"]:
                sample = ", ".join(f"`{sig}`" for sig in metrics["repeated_signatures"][:6])
                lines.append(f"- repeated windows: {sample}")
            lines.append("")
    else:
        lines.append("_No strongly patternized fragments met the flag threshold._")
        lines.append("")

    lines.append("## Fragment Hotspots")
    lines.append("")
    if not rows:
        lines.append("_No fragment hotspots met the threshold._")
        lines.append("")
    else:
        for rank, row in enumerate(rows[:top], 1):
            frag = row["fragment"]
            lines.append(f"### {rank}. `{frag.section}` ({row['match_count']} near matches, {row['hint']})")
            lines.append(f"- shape: reusable={row['reusable']}, unique={row['unique']}, reuse={row['reusable_ratio']:.2f}")
            lines.append(f"- fragment: {snippet(frag.text)}")
            for match in row["matches"][:5]:
                other = fragments[match["idx"]]
                m = match["metrics"]
                lines.append(
                    f"- near: raw={m['raw_overlap']:.2f} seq={m['raw_seq']:.2f} "
                    f"content={m['content_overlap']:.2f} bi={m['bigram']:.2f}: {snippet(other.text, 150)}"
                )
            lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def pattern_model_from_template(row: dict | None) -> dict | None:
    if not row:
        return None
    positions = row["positions"]
    return {
        "length": len(positions),
        "positions": positions,
        "soft_text": soft_template_text(positions),
        "source_count": row["count"],
        "sections": row["sections"],
        "examples": row["examples"],
    }


def score_window_against_model(tokens: tuple[str, ...], model: dict) -> tuple[float, int, float]:
    probabilities: list[float] = []
    matched = 0
    for token, counter in zip(tokens, model["positions"]):
        total = sum(counter.values())
        probability = counter[token] / total if total else 0.0
        probabilities.append(probability)
        if probability > 0:
            matched += 1
    score = sum(probabilities) / max(1, len(probabilities))
    peak = max(probabilities) if probabilities else 0.0
    return score, matched, peak


def pattern_candidates(
    fragments: list[Fragment],
    model: dict,
    *,
    min_score: float,
) -> list[dict]:
    candidates_by_fragment: dict[int, dict] = {}
    length = model["length"]
    min_matched = max(3, int(length * 0.68))
    for frag in fragments:
        if len(frag.tokens) < length:
            continue
        best: dict | None = None
        for start in range(0, len(frag.tokens) - length + 1):
            window = frag.tokens[start : start + length]
            score, matched, peak = score_window_against_model(window, model)
            if matched < min_matched or score < min_score:
                continue
            candidate = {
                "fragment": frag,
                "window": window,
                "start": start,
                "score": score,
                "matched": matched,
                "peak": peak,
            }
            if not best or (candidate["score"], candidate["matched"], -candidate["start"]) > (
                best["score"],
                best["matched"],
                -best["start"],
            ):
                best = candidate
        if best:
            candidates_by_fragment[frag.idx] = best
    return sorted(candidates_by_fragment.values(), key=lambda item: (item["fragment"].para_index, item["fragment"].idx))


def select_pattern_candidates(candidates: list[dict], policy: str) -> list[dict]:
    by_part: dict[int, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_part[candidate["fragment"].para_index].append(candidate)

    selected: list[dict] = []
    if policy.startswith("floor"):
        keep = int(policy.removeprefix("floor"))
        for group in by_part.values():
            group.sort(key=lambda item: item["fragment"].idx)
            selected.extend(group[keep:])
        return selected

    if policy == "ceiling":
        scores = sorted(candidate["score"] for candidate in candidates)
        if not scores:
            return []
        threshold = max(0.58, scores[int(len(scores) * 0.72)])
        for group in by_part.values():
            high_confidence = [candidate for candidate in group if candidate["score"] >= threshold]
            high_confidence.sort(key=lambda item: item["fragment"].idx)
            selected.extend(high_confidence[1:])
        return selected

    raise ValueError(f"Unknown pattern policy: {policy}")


def write_pattern_bakeoff(
    text: str,
    path: Path,
    fragments: list[Fragment],
    template_rows: list[dict],
    out_dir: Path,
    *,
    min_score: float,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = pattern_model_from_template(template_rows[0] if template_rows else None)
    if not model:
        return []

    candidates = pattern_candidates(fragments, model, min_score=min_score)
    original_words = word_count(text)
    written: list[dict] = []
    policies = ("floor1", "floor2", "ceiling")
    for policy in policies:
        selected = select_pattern_candidates(candidates, policy)
        rows = [{"fragment": candidate["fragment"]} for candidate in selected]
        output = apply_profile(text, rows)
        kept_words = word_count(output)
        out_path = out_dir / f"{path.stem}.pattern-{policy}.md"
        out_path.write_text(output, encoding="utf-8")
        written.append(
            {
                "policy": policy,
                "path": out_path,
                "candidate_count": len(candidates),
                "removed_fragments": len({candidate["fragment"].idx for candidate in selected}),
                "original_words": original_words,
                "kept_words": kept_words,
                "stripped_words": max(0, original_words - kept_words),
                "marker_count": output.count("@"),
            }
        )

    report_path = out_dir / f"{path.stem}.pattern-bakeoff.md"
    lines = [
        f"# Pattern Bakeoff: {path.name}",
        "",
        f"- model: `{model['soft_text']}`",
        f"- source template hits: {model['source_count']}",
        f"- candidate fragments: {len(candidates)}",
        f"- match threshold: {min_score:.2f}",
        "",
        "## Position Probabilities",
        "",
    ]
    for index, counter in enumerate(model["positions"], 1):
        total = sum(counter.values())
        values = ", ".join(
            f"`{token}` {count / total:.2%} ({count})"
            for token, count in counter.most_common(10)
        )
        lines.append(f"- {index}: {values}")
    lines.extend(["", "## Outputs", ""])
    for row in written:
        lines.append(
            f"- `{row['policy']}`: removed {row['removed_fragments']} fragments, "
            f"stripped {row['stripped_words']} words, markers {row['marker_count']} -> `{row['path']}`"
        )
    lines.extend(["", "## Top Candidate Examples", ""])
    for candidate in sorted(candidates, key=lambda item: (-item["score"], item["fragment"].idx))[:20]:
        lines.append(
            f"- score={candidate['score']:.3f}, matched={candidate['matched']}, "
            f"`{candidate['fragment'].section}`: {snippet(candidate['fragment'].text, 180)}"
        )
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    written.append({"policy": "report", "path": report_path})
    return written


def snippet(text: str, limit: int = 260) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "..."


def resolve_book_path(args: argparse.Namespace) -> Path:
    if bool(args.book) == bool(args.file):
        raise SystemExit("Provide exactly one of --book or --file")
    path = BOOKS_DIR / f"{args.book}.md" if args.book else Path(args.file)
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return path


def parse_profile_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise SystemExit("--profiles must name at least one profile")
    unknown = [name for name in names if name not in REFLOOR_PROFILES]
    if unknown:
        allowed = ", ".join(REFLOOR_PROFILES)
        raise SystemExit(f"Unknown profile(s): {', '.join(unknown)}. Allowed: {allowed}")
    return names


def frontier_growth_for_args(args: argparse.Namespace) -> int:
    if args.frontier_growth > 0:
        return args.frontier_growth
    first_profile = REFLOOR_PROFILES[args.profile_names[0]]
    return first_profile["frontier_growth"]


def analyze_book(path: Path, args: argparse.Namespace, out_dir: Path | None = None) -> dict:
    text = path.read_text(encoding="utf-8")
    frontmatter, _ = strip_frontmatter(text)
    paragraphs = parse_paragraphs(text)
    character_tokens = extract_character_tokens(frontmatter) | extract_speaker_tokens(text)
    fragments = build_fragments(paragraphs, min_tokens=args.min_tokens, excluded_tokens=character_tokens)
    rows = score_fragments(
        fragments,
        paragraph_radius=args.paragraph_radius,
        frontier_growth=frontier_growth_for_args(args),
        min_match_score=args.min_match_score,
        min_shared_content=args.min_shared_content,
        max_candidates_per_token=args.max_candidates_per_token,
        excluded_tokens=character_tokens,
    )
    template_rows = gapped_template_rows(fragments, max_items=args.template_top)
    phrase_rows = repeated_phrases(rows)
    pattern_rows = flag_patternized_fragments(fragments)

    written: list[dict] = []
    bakeoff_written: list[dict] = []
    report_path: Path | None = None
    if out_dir:
        written = write_refloors(text, path, rows, fragments, out_dir, args.profile_names)
        report_path = write_pattern_report(
            path,
            rows,
            fragments,
            template_rows,
            phrase_rows,
            pattern_rows,
            out_dir,
            top=args.top,
        )
    if args.write_pattern_bakeoff:
        bakeoff_written = write_pattern_bakeoff(
            text,
            path,
            fragments,
            template_rows,
            Path(args.write_pattern_bakeoff),
            min_score=args.pattern_min_score,
        )

    return {
        "path": path,
        "paragraphs": len(paragraphs),
        "fragments": len(fragments),
        "fragment_items": fragments,
        "rows": rows,
        "template_rows": template_rows,
        "phrase_rows": phrase_rows,
        "pattern_rows": pattern_rows,
        "written": written,
        "bakeoff_written": bakeoff_written,
        "report_path": report_path,
    }


def write_metrics_summary(results: list[dict], out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "refloor-word-metrics.csv"
    md_path = out_dir / "refloor-word-metrics.md"
    csv_lines = [
        "book,profile,original_words,kept_words,stripped_words,kept_pct,removed_fragments,fragments,pattern_flags"
    ]
    md_lines = [
        "# Refloor Word Metrics",
        "",
        "| Book | Profile | Original | Kept | Stripped | Kept % | Removed fragments | Pattern flags |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in sorted(results, key=lambda item: item["path"].name):
        book = result["path"].stem
        for row in result["written"]:
            csv_lines.append(
                ",".join(
                    [
                        book,
                        row["profile"],
                        str(row["original_words"]),
                        str(row["kept_words"]),
                        str(row["stripped_words"]),
                        f"{row['kept_pct']:.2f}",
                        str(row["removed_fragments"]),
                        str(result["fragments"]),
                        str(len(result["pattern_rows"])),
                    ]
                )
            )
            md_lines.append(
                f"| `{book}` | `{row['profile']}` | {row['original_words']} | {row['kept_words']} | "
                f"{row['stripped_words']} | {row['kept_pct']:.2f} | {row['removed_fragments']} | "
                f"{len(result['pattern_rows'])} |"
            )

    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Find local fragment repetition hotspots in a markdown document")
    parser.add_argument("--book", help="Document slug under the content directory")
    parser.add_argument("--file", help="Explicit markdown file path")
    parser.add_argument("--all", action="store_true", help="Process every *.md file in the content directory")
    parser.add_argument("--top", type=int, default=40, help="How many ranked fragments to print")
    parser.add_argument("--template-top", type=int, default=40, help="How many diffuse scaffold templates to report")
    parser.add_argument("--profiles", default="med", help="Comma-separated refloor profiles to write; default: med")
    parser.add_argument("--paragraph-radius", type=int, default=6, help="Compare fragments within this many paragraphs")
    parser.add_argument("--frontier-growth", type=int, default=0, help="How much the search radius grows per frontier expansion. 0 = auto (use profile default: low=1, med=3, high=4). 1 = fixed radius. Higher = reach further for repeating patterns.")
    parser.add_argument("--min-tokens", type=int, default=3, help="Minimum raw tokens per fragment")
    parser.add_argument("--min-match-score", type=float, default=0.24, help="Minimum fragment similarity to count as a near-match")
    parser.add_argument("--min-shared-content", type=int, default=1, help="Minimum shared non-stopword tokens unless raw bigrams match")
    parser.add_argument("--max-candidates-per-token", type=int, default=180, help="Ignore token postings larger than this")
    parser.add_argument("--write-refloors", help="Directory to write selected refloored markdown outputs")
    parser.add_argument("--write-pattern-bakeoff", help="Directory to write soft-pattern bakeoff outputs")
    parser.add_argument("--pattern-min-score", type=float, default=0.52, help="Minimum soft-pattern probability score for bakeoff candidates")
    args = parser.parse_args()
    args.profile_names = parse_profile_names(args.profiles)

    if args.all:
        if args.book or args.file:
            raise SystemExit("--all cannot be combined with --book or --file")
        if not args.write_refloors:
            raise SystemExit("--all requires --write-refloors")
        if args.write_pattern_bakeoff:
            raise SystemExit("--write-pattern-bakeoff is only supported for single-book runs")
        out_dir = Path(args.write_refloors)
        paths = sorted(BOOKS_DIR.glob("*.md"))
        results = []
        for index, path in enumerate(paths, 1):
            print(f"[{index}/{len(paths)}] {path.name}")
            results.append(analyze_book(path, args, out_dir))
        csv_path, md_path = write_metrics_summary(results, out_dir)
        print()
        print(f"Wrote metrics: `{csv_path}`")
        print(f"Wrote metrics: `{md_path}`")
        return

    path = resolve_book_path(args)
    out_dir = Path(args.write_refloors) if args.write_refloors else None
    result = analyze_book(path, args, out_dir)
    rows = result["rows"]
    fragments = result["fragment_items"]
    template_rows = result["template_rows"]
    phrase_rows = result["phrase_rows"]
    pattern_rows = result["pattern_rows"]

    print(f"# Repetition Hotspots: {path.name}")
    print()
    print(f"- paragraphs: {result['paragraphs']}")
    print(f"- fragments: {result['fragments']}")
    print(f"- fragments with local near-matches: {len(rows)}")
    print(f"- paragraph radius: {args.paragraph_radius}")
    print(f"- match threshold: {args.min_match_score:.2f}")
    print()

    if args.write_refloors:
        print("## Refloored Outputs")
        print()
        for row in result["written"]:
            print(
                f"- `{row['profile']}`: removed {row['removed_fragments']} fragments, "
                f"stripped {row['stripped_words']} words -> `{row['path']}`"
            )
        print(f"- `inside-fragments`: wrote analysis report -> `{result['report_path']}`")
        print()
    if args.write_pattern_bakeoff:
        print("## Pattern Bakeoff Outputs")
        print()
        for row in result["bakeoff_written"]:
            if row["policy"] == "report":
                print(f"- `report`: `{row['path']}`")
                continue
            print(
                f"- `{row['policy']}`: removed {row['removed_fragments']} fragments, "
                f"stripped {row['stripped_words']} words, markers {row['marker_count']} -> `{row['path']}`"
            )
        print()

    print("## Diffuse Scaffold Templates")
    print()
    if template_rows:
        for rank, row in enumerate(template_rows, 1):
            print(
                f"### {rank}. `{row['soft_text']}` "
                f"({row['count']} hits, {row['position_variety']} positional variants, {len(row['sections'])} sections)"
            )
            for idx, position in enumerate(row["positions"], 1):
                values = ", ".join(f"`{tok}` ({count})" for tok, count in position.most_common(5))
                print(f"- position {idx}: {values}")
            if row["examples"]:
                print("- examples: " + "; ".join(f"`{example}`" for example in row["examples"]))
            print()
    else:
        print("_No diffuse scaffold templates met the threshold._")
        print()

    print("## Short Phrase Diagnostics")
    print()
    if phrase_rows:
        print(", ".join(f"`{phrase}` ({count})" for phrase, count in phrase_rows))
    else:
        print("_No repeated local phrase patterns met the threshold._")
    print()

    print("## Inside-Fragment Pattern Flags")
    print()
    if pattern_rows:
        for rank, row in enumerate(pattern_rows[: args.top], 1):
            frag = row["fragment"]
            metrics = row["metrics"]
            reasons = ", ".join(metrics["reasons"])
            print(
                f"### {rank}. score {metrics['pattern_score']:.3f} - `{frag.section}` L{frag.start_line} "
                f"(repeat={metrics['repeat_density']:.2f}, reuse={metrics['repeated_mass']}, "
                f"diversity={metrics['uniqueness_ratio']:.2f}, dominant={metrics['dominant_share']:.2f}, {reasons})"
            )
            print(f"- fragment: {snippet(frag.text)}")
            if metrics["repeated_signatures"]:
                sample = ", ".join(f"`{sig}`" for sig in metrics["repeated_signatures"][:6])
                print(f"- repeated windows: {sample}")
            print()
    else:
        print("_No strongly patternized fragments met the flag threshold._")
        print()

    print("## Fragment Hotspots")
    print()
    if not rows:
        print("_No fragment hotspots met the threshold._")
        return

    for rank, row in enumerate(rows[: args.top], 1):
        frag = row["fragment"]
        print(
            f"### {rank}. score {row['badness']:.3f} - `{frag.section}` L{frag.start_line} "
            f"({row['match_count']} matches, reusable={row['reusable']}, unique={row['unique']}, "
            f"reuse={row['reusable_ratio']:.2f}, hint={row['hint']})"
        )
        print(f"- fragment: {snippet(frag.text)}")
        for match in row["matches"][:5]:
            other = fragments[match["idx"]]
            m = match["metrics"]
            print(
                f"- match L{other.start_line} d={match['distance']} score={m['score']:.2f} "
                f"raw={m['raw_overlap']:.2f} seq={m['raw_seq']:.2f} "
                f"content={m['content_overlap']:.2f} bi={m['bigram']:.2f}: {snippet(other.text, 150)}"
            )
        print()


if __name__ == "__main__":
    main()
