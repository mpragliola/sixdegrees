/**
 * Standalone Okapi BM25 lexical search index. Independent pathway from the
 * embedding-based search — not bolted onto it. Consumers wire it in via the
 * search orchestrator's `similarity: "bm25"` option.
 */

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0);
}

interface DocEntry {
  id: string;
  termCounts: Map<string, number>;
  length: number;
}

export interface Bm25Options {
  /** Term-frequency saturation. Default 1.2. */
  k1?: number;
  /** Length-normalization strength (0 = none, 1 = full). Default 0.75. */
  b?: number;
}

export class Bm25Index {
  private docs: DocEntry[] = [];
  private idf: Map<string, number> = new Map();
  private avgDocLength = 0;
  private readonly k1: number;
  private readonly b: number;

  constructor(docs: { id: string; text: string }[], options: Bm25Options = {}) {
    this.k1 = options.k1 ?? 1.2;
    this.b = options.b ?? 0.75;

    const df = new Map<string, number>();
    let totalLength = 0;
    for (const doc of docs) {
      const tokens = tokenize(doc.text);
      const termCounts = new Map<string, number>();
      for (const tok of tokens) {
        termCounts.set(tok, (termCounts.get(tok) ?? 0) + 1);
      }
      this.docs.push({ id: doc.id, termCounts, length: tokens.length });
      totalLength += tokens.length;
      for (const term of termCounts.keys()) {
        df.set(term, (df.get(term) ?? 0) + 1);
      }
    }

    const n = this.docs.length;
    this.avgDocLength = n > 0 ? totalLength / n : 0;
    for (const [term, dfCount] of df.entries()) {
      // BM25 idf: ln(1 + (N - df + 0.5) / (df + 0.5)) — always positive.
      this.idf.set(term, Math.log(1 + (n - dfCount + 0.5) / (dfCount + 0.5)));
    }
  }

  search(query: string, topK = 10): { id: string; score: number }[] {
    const tokens = tokenize(query);
    if (tokens.length === 0) return [];

    const queryCounts = new Map<string, number>();
    for (const tok of tokens) {
      queryCounts.set(tok, (queryCounts.get(tok) ?? 0) + 1);
    }

    const { k1, b, avgDocLength } = this;
    const results: { id: string; score: number }[] = [];
    for (const doc of this.docs) {
      let score = 0;
      const lengthNorm =
        avgDocLength > 0 ? 1 - b + (b * doc.length) / avgDocLength : 1;
      for (const [term, qCount] of queryCounts.entries()) {
        const tf = doc.termCounts.get(term);
        if (tf === undefined) continue;
        const idf = this.idf.get(term) ?? 0;
        score += qCount * idf * ((tf * (k1 + 1)) / (tf + k1 * lengthNorm));
      }
      results.push({ id: doc.id, score });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }
}
