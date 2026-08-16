/**
 * Standalone TF-IDF lexical search index. Independent pathway from the
 * embedding-based search — not bolted onto it. Consumers wire it in via the
 * search orchestrator's `similarity: "tfidf"` option.
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

export class TfIdfIndex {
  private docs: DocEntry[] = [];
  private df: Map<string, number> = new Map();
  private idf: Map<string, number> = new Map();
  private docVectors: Map<string, Map<string, number>> = new Map();
  private docNorms: Map<string, number> = new Map();

  constructor(docs: { id: string; text: string }[]) {
    for (const doc of docs) {
      const tokens = tokenize(doc.text);
      const termCounts = new Map<string, number>();
      for (const tok of tokens) {
        termCounts.set(tok, (termCounts.get(tok) ?? 0) + 1);
      }
      this.docs.push({ id: doc.id, termCounts, length: tokens.length });
      for (const term of termCounts.keys()) {
        this.df.set(term, (this.df.get(term) ?? 0) + 1);
      }
    }

    const n = this.docs.length;
    for (const [term, df] of this.df.entries()) {
      // Standard smoothed idf: log(N / df) + 1, floor at 0.
      const idf = Math.log(n / df) + 1;
      this.idf.set(term, idf);
    }

    for (const doc of this.docs) {
      const vec = new Map<string, number>();
      for (const [term, count] of doc.termCounts.entries()) {
        const tf = count / Math.max(1, doc.length);
        const idf = this.idf.get(term) ?? 0;
        vec.set(term, tf * idf);
      }
      let normSq = 0;
      for (const w of vec.values()) normSq += w * w;
      this.docVectors.set(doc.id, vec);
      this.docNorms.set(doc.id, Math.sqrt(normSq));
    }
  }

  search(query: string, topK = 10): { id: string; score: number }[] {
    const tokens = tokenize(query);
    if (tokens.length === 0) return [];

    const queryCounts = new Map<string, number>();
    for (const tok of tokens) {
      queryCounts.set(tok, (queryCounts.get(tok) ?? 0) + 1);
    }

    const queryVec = new Map<string, number>();
    for (const [term, count] of queryCounts.entries()) {
      const tf = count / tokens.length;
      const idf = this.idf.get(term) ?? 0;
      queryVec.set(term, tf * idf);
    }
    let queryNormSq = 0;
    for (const w of queryVec.values()) queryNormSq += w * w;
    const queryNorm = Math.sqrt(queryNormSq);

    const results: { id: string; score: number }[] = [];
    for (const doc of this.docs) {
      const docVec = this.docVectors.get(doc.id)!;
      const docNorm = this.docNorms.get(doc.id) ?? 0;
      if (docNorm === 0 || queryNorm === 0) {
        results.push({ id: doc.id, score: 0 });
        continue;
      }
      let dot = 0;
      for (const [term, qWeight] of queryVec.entries()) {
        const dWeight = docVec.get(term);
        if (dWeight !== undefined) dot += qWeight * dWeight;
      }
      const score = dot / (queryNorm * docNorm);
      results.push({ id: doc.id, score });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }
}
