/**
 * Builds inline highlight markup for a note's title and body, showing
 * per-character match strength (max score across overlapping chunk layers)
 * plus exact query-term matches, via a boundary-sweep merge of all
 * contributing ranges into non-overlapping segments.
 */
import { removeStopwords, eng, fra, deu, spa, ita, por, nld } from "stopword";
import type { ScoredChunk } from "./chunkIndex.js";

interface Range {
  start: number;
  end: number;
  score: number;
}

// A handful of major-language stopword lists (not all 62 supported by the
// package) so exact-match highlighting skips filler words ("the", "de",
// "und", ...) across the languages demo notes are realistically written in,
// without bloating the query-term filter with rarely-needed lists.
const STOPWORDS = [...eng, ...fra, ...deu, ...spa, ...ita, ...por, ...nld];

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Splits a query into distinct, non-empty lowercase terms for exact
 * matching, dropping stopwords (filler words like "the", "de", "und") so
 * they don't get bold/underline treatment alongside meaningful terms.
 */
function queryTerms(query: string): string[] {
  const terms = query
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0);
  const meaningful = removeStopwords(terms, STOPWORDS);
  return Array.from(new Set(meaningful.map((t) => t.toLowerCase())));
}

/** Finds all case-insensitive occurrences of any query term in `text`. */
function findExactMatches(text: string, terms: string[]): { start: number; end: number }[] {
  if (terms.length === 0) return [];
  const pattern = terms.map(escapeRegExp).join("|");
  const re = new RegExp(pattern, "gi");
  const matches: { start: number; end: number }[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m[0].length === 0) {
      re.lastIndex++;
      continue;
    }
    matches.push({ start: m.index, end: m.index + m[0].length });
  }
  return matches;
}

/**
 * Merges overlapping score ranges into non-overlapping segments via a
 * boundary-sweep: collect all range endpoints, sort/dedupe, then compute the
 * max score covering each consecutive boundary pair.
 */
function mergeScoreRanges(text: string, ranges: Range[]): { start: number; end: number; score: number }[] {
  if (ranges.length === 0) return [{ start: 0, end: text.length, score: 0 }];

  const boundaries = new Set<number>([0, text.length]);
  for (const r of ranges) {
    boundaries.add(Math.max(0, Math.min(r.start, text.length)));
    boundaries.add(Math.max(0, Math.min(r.end, text.length)));
  }
  const sorted = Array.from(boundaries).sort((a, b) => a - b);

  const segments: { start: number; end: number; score: number }[] = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const start = sorted[i]!;
    const end = sorted[i + 1]!;
    if (start >= end) continue;
    let score = 0;
    for (const r of ranges) {
      if (r.start <= start && r.end >= end) {
        score = Math.max(score, r.score);
      }
    }
    segments.push({ start, end, score });
  }
  return segments;
}

const MIN_VISIBLE_ALPHA = 0.08;
const MAX_ALPHA = 0.75;

/**
 * Maps a raw score to a background alpha by its position within [minScore,
 * maxScore] (the range of scores contributing to this note), so the
 * strongest-matching segment in a note always reads at full intensity even
 * when absolute scores (e.g. cosine similarity) cluster tightly. Any
 * positive score gets at least MIN_VISIBLE_ALPHA so weak matches stay
 * perceptible rather than rounding to invisible.
 */
function scoreToAlpha(score: number, minScore: number, maxScore: number): number {
  if (score <= 0) return 0;
  if (maxScore <= minScore) return MAX_ALPHA;
  const normalized = (score - minScore) / (maxScore - minScore);
  return MIN_VISIBLE_ALPHA + normalized * (MAX_ALPHA - MIN_VISIBLE_ALPHA);
}

/**
 * Renders `text` as HTML with per-segment background intensity (a
 * continuous alpha normalized against [minScore, maxScore] for the note,
 * applied via an inline --hl-alpha custom property) and bold+underline
 * exact query-term matches, escaping raw text safely without corrupting
 * char offsets.
 */
function renderHighlighted(
  text: string,
  scoreRanges: Range[],
  query: string,
  minScore: number,
  maxScore: number,
): string {
  const scoreSegments = mergeScoreRanges(text, scoreRanges);
  const exactMatches = findExactMatches(text, queryTerms(query));

  // Merge in exact-match boundaries too, so bold/underline can be applied
  // per-segment without crossing tag boundaries.
  const boundaries = new Set<number>([0, text.length]);
  for (const s of scoreSegments) {
    boundaries.add(s.start);
    boundaries.add(s.end);
  }
  for (const m of exactMatches) {
    boundaries.add(Math.max(0, Math.min(m.start, text.length)));
    boundaries.add(Math.max(0, Math.min(m.end, text.length)));
  }
  const sorted = Array.from(boundaries).sort((a, b) => a - b);

  let html = "";
  for (let i = 0; i < sorted.length - 1; i++) {
    const start = sorted[i]!;
    const end = sorted[i + 1]!;
    if (start >= end) continue;

    const seg = scoreSegments.find((s) => s.start <= start && s.end >= end);
    const alpha = seg ? scoreToAlpha(seg.score, minScore, maxScore) : 0;
    const isExact = exactMatches.some((m) => m.start <= start && m.end >= end);

    const chunk = escapeHtml(text.slice(start, end));
    const classes = [alpha > 0 ? "hl" : "", isExact ? "hl-exact" : ""].filter(Boolean).join(" ");
    const style = alpha > 0 ? ` style="--hl-alpha: ${alpha.toFixed(3)}"` : "";
    html += classes ? `<span class="${classes}"${style}>${chunk}</span>` : chunk;
  }
  return html;
}

export interface HighlightedNote {
  titleHtml: string;
  bodyHtml: string;
}

/**
 * Computes the (min, max) chunk score across an entire result set, so every
 * note's highlight intensity is normalized against the same scale instead of
 * each note stretching its own weakest match to full brightness.
 */
export function globalScoreRange(allMatchedChunks: ScoredChunk[][]): { minScore: number; maxScore: number } {
  let minScore = Infinity;
  let maxScore = -Infinity;
  for (const chunks of allMatchedChunks) {
    for (const { score } of chunks) {
      if (score <= 0) continue;
      if (score < minScore) minScore = score;
      if (score > maxScore) maxScore = score;
    }
  }
  if (!Number.isFinite(minScore)) minScore = 0;
  if (!Number.isFinite(maxScore)) maxScore = 0;
  return { minScore, maxScore };
}

/**
 * Builds highlighted title/body HTML for a note from its matched chunks.
 * Handles the three chunk coordinate spaces:
 *  - "title" layer: offsets into note.title directly.
 *  - "note" (whole-note) layer: offsets into `${title}\n\n${body}`; split
 *    and re-mapped into title-range + body-range contributions.
 *  - "paragraph" / "sentence-window": offsets into note.body directly.
 *
 * `scoreRange` should come from globalScoreRange() over the full result set
 * so intensity is comparable across notes, not just within one.
 */
export function buildHighlightedNote(
  note: { title: string; body: string },
  matchedChunks: ScoredChunk[],
  query: string,
  scoreRange: { minScore: number; maxScore: number },
): HighlightedNote {
  const titleRanges: Range[] = [];
  const bodyRanges: Range[] = [];
  const separatorLen = 2; // "\n\n"

  for (const { chunk, score } of matchedChunks) {
    if (score <= 0) continue;
    if (chunk.layer === "title") {
      titleRanges.push({ start: chunk.charStart, end: chunk.charEnd, score });
    } else if (chunk.layer === "paragraph" || chunk.layer === "sentence-window") {
      bodyRanges.push({ start: chunk.charStart, end: chunk.charEnd, score });
    } else if (chunk.layer === "note") {
      const titleEnd = note.title.length;
      const bodyStart = titleEnd + separatorLen;
      if (chunk.charStart < titleEnd) {
        titleRanges.push({ start: chunk.charStart, end: Math.min(chunk.charEnd, titleEnd), score });
      }
      if (chunk.charEnd > bodyStart) {
        bodyRanges.push({
          start: Math.max(chunk.charStart, bodyStart) - bodyStart,
          end: chunk.charEnd - bodyStart,
          score,
        });
      }
    }
  }

  const { minScore, maxScore } = scoreRange;

  return {
    titleHtml: renderHighlighted(note.title, titleRanges, query, minScore, maxScore),
    bodyHtml: renderHighlighted(note.body, bodyRanges, query, minScore, maxScore),
  };
}
