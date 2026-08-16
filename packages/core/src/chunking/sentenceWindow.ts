import type { Chunk, ChunkingStrategy } from "../types.js";

/**
 * Naive sentence splitter: splits on `.`, `!`, `?` followed by whitespace.
 * Does NOT guard against abbreviations (e.g. "Dr. Smith" splits incorrectly).
 * This is a deliberate simplification — see spec: "naive is fine, document
 * the limitation."
 */
function splitSentences(text: string): { text: string; start: number; end: number }[] {
  const sentences: { text: string; start: number; end: number }[] = [];
  const re = /[^.!?]+[.!?]+(\s+|$)|[^.!?]+$/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    const raw = match[0];
    const trimmed = raw.trim();
    if (trimmed.length === 0) continue;
    const start = match.index + raw.indexOf(trimmed);
    sentences.push({ text: trimmed, start, end: start + trimmed.length });
  }
  return sentences;
}

export interface SentenceWindowOptions {
  /** Number of sentences per window. Default 3. */
  windowSize?: number;
  /** Number of sentences of overlap between consecutive windows. Default 1. */
  overlap?: number;
}

/**
 * Sliding window over sentences (regex-based splitting, naive — see
 * splitSentences). Layer "sentence-window".
 */
export class SentenceWindowChunker implements ChunkingStrategy {
  readonly name = "sentence-window";
  private readonly windowSize: number;
  private readonly overlap: number;

  constructor(options: SentenceWindowOptions = {}) {
    this.windowSize = options.windowSize ?? 3;
    this.overlap = options.overlap ?? 1;
    if (this.windowSize <= 0) {
      throw new Error("windowSize must be > 0");
    }
    if (this.overlap < 0 || this.overlap >= this.windowSize) {
      throw new Error("overlap must be >= 0 and < windowSize");
    }
  }

  chunk(note: { id: string; title: string; body: string }): Chunk[] {
    const sentences = splitSentences(note.body);
    if (sentences.length === 0) return [];

    const step = this.windowSize - this.overlap;
    const chunks: Chunk[] = [];
    let windowIndex = 0;

    for (let i = 0; i < sentences.length; i += step) {
      const windowSentences = sentences.slice(i, i + this.windowSize);
      if (windowSentences.length === 0) break;

      const first = windowSentences[0]!;
      const last = windowSentences[windowSentences.length - 1]!;
      const text = note.body.slice(first.start, last.end).trim();

      chunks.push({
        id: `${note.id}::sentence-window::${windowIndex}`,
        noteId: note.id,
        layer: "sentence-window",
        text,
        charStart: first.start,
        charEnd: last.end,
      });
      windowIndex++;

      if (i + this.windowSize >= sentences.length) break;
    }

    return chunks;
  }
}
