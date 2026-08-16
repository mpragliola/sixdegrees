/**
 * A thin lab-side index built directly on sixdegrees' exported primitives
 * (chunkers, EmbeddingProvider, similarity fns, TfIdfIndex, fuseChunkScores)
 * rather than InMemorySearchIndex. We need this because InMemorySearchIndex
 * returns matchedChunks: Chunk[] without attaching per-chunk scores, which
 * we need for the "per-layer match badge" UI (spec: "small badges showing
 * 'paragraph: 0.82', 'title: 0.65'"). Re-implementing the score plumbing
 * here (using the same exported building blocks core uses internally) keeps
 * packages/core untouched while giving the lab exactly what it needs.
 */
import {
  cosineSimilarity,
  dotProduct,
  euclideanSimilarity,
  fuseChunkScores,
  TfIdfIndex,
  type Chunk,
  type ChunkingStrategy,
  type EmbeddingProvider,
  type FusionConfig,
  type Note,
  type SimilarityFn,
} from "sixdegrees";

export type SimilarityMetric = "cosine" | "dot" | "euclidean" | "tfidf";

export interface ScoredChunk {
  chunk: Chunk;
  score: number;
}

export interface LabSearchResult {
  noteId: string;
  score: number;
  /** Best-scoring chunk per layer, descending by score. */
  matchedChunks: ScoredChunk[];
}

const SIMILARITY_FNS: Record<Exclude<SimilarityMetric, "tfidf">, SimilarityFn> = {
  cosine: cosineSimilarity,
  dot: dotProduct,
  euclidean: euclideanSimilarity,
};

/**
 * transformers.js pads every sequence in a single embed() call to the length
 * of the longest one in that call. The layered strategy mixes tiny title
 * chunks with full long-note chunks in the same corpus, so a single
 * unbatched embed() call over all chunks can blow up the effective tensor
 * size (observed as a WASM `std::bad_alloc` / OrtRun failure in-browser).
 * Batching bounds both the call size and the worst-case padding blowup.
 */
const EMBED_BATCH_SIZE = 16;

async function embedBatched(
  embedder: EmbeddingProvider,
  texts: string[],
  onProgress?: (done: number, total: number) => void,
): Promise<Float32Array[]> {
  const out: Float32Array[] = [];
  onProgress?.(0, texts.length);
  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    const batch = texts.slice(i, i + EMBED_BATCH_SIZE);
    out.push(...(await embedder.embed(batch)));
    onProgress?.(Math.min(i + EMBED_BATCH_SIZE, texts.length), texts.length);
    // Yield to the event loop between batches so the UI (spinner/progress
    // text) can actually repaint — embedding runs on the main thread.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return out;
}

export class LabChunkIndex {
  private chunks: Chunk[] = [];
  private embeddings: Float32Array[] = [];
  private tfidf: TfIdfIndex | null = null;
  private embedder: EmbeddingProvider | null = null;

  async build(
    notes: Note[],
    chunker: ChunkingStrategy,
    embedder: EmbeddingProvider,
    onProgress?: (done: number, total: number) => void,
  ): Promise<void> {
    const chunks: Chunk[] = [];
    for (const note of notes) {
      chunks.push(...chunker.chunk(note));
    }
    this.chunks = chunks;
    this.embedder = embedder;
    this.embeddings =
      chunks.length > 0 ? await embedBatched(embedder, chunks.map((c) => c.text), onProgress) : [];
    this.tfidf = new TfIdfIndex(chunks.map((c) => ({ id: c.id, text: c.text })));
  }

  async search(query: string, opts: { similarity: SimilarityMetric; fusion: FusionConfig; topK?: number }): Promise<LabSearchResult[]> {
    const topK = opts.topK ?? 10;
    const scoreByChunkId = new Map<string, number>();

    if (opts.similarity === "tfidf") {
      if (!this.tfidf) throw new Error("Index not built.");
      const tfidfResults = this.tfidf.search(query, this.chunks.length || 1);
      for (const r of tfidfResults) scoreByChunkId.set(r.id, r.score);
    } else {
      if (!this.embedder) throw new Error("Index not built.");
      const simFn = SIMILARITY_FNS[opts.similarity];
      const [queryEmbedding] = await this.embedder.embed([query]);
      if (queryEmbedding) {
        this.chunks.forEach((chunk, i) => {
          const emb = this.embeddings[i];
          if (!emb) return;
          scoreByChunkId.set(chunk.id, simFn(queryEmbedding, emb));
        });
      }
    }

    const chunkScores = this.chunks
      .map((c) => ({ noteId: c.noteId, layer: c.layer, score: scoreByChunkId.get(c.id) ?? 0 }))
      .filter((cs) => cs.score > 0);

    const noteScores = fuseChunkScores(chunkScores, opts.fusion);

    return noteScores.slice(0, topK).map((ns) => {
      const noteChunks = this.chunks
        .filter((c) => c.noteId === ns.noteId)
        .map((c) => ({ chunk: c, score: scoreByChunkId.get(c.id) ?? 0 }))
        .filter((sc) => sc.score > 0)
        .sort((a, b) => b.score - a.score);
      return { noteId: ns.noteId, score: ns.score, matchedChunks: noteChunks };
    });
  }
}
