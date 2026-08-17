/**
 * A thin lab-side index built directly on sixdegrees' exported primitives
 * (chunkers, EmbeddingProvider, similarity fns, Bm25Index, fuseChunkScores)
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
  Bm25Index,
  type Chunk,
  type ChunkingStrategy,
  type EmbeddingProvider,
  type FusionConfig,
  type Note,
  type SimilarityFn,
} from "sixdegrees";
import { log } from "./log.js";

export type SimilarityMetric = "cosine" | "dot" | "euclidean" | "bm25";

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

const SIMILARITY_FNS: Record<Exclude<SimilarityMetric, "bm25">, SimilarityFn> = {
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
  log("embed", `embedBatched: ${texts.length} texts, batch size ${EMBED_BATCH_SIZE}`);
  const batchStart = performance.now();
  const out: Float32Array[] = [];
  onProgress?.(0, texts.length);
  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    const batch = texts.slice(i, i + EMBED_BATCH_SIZE);
    const t0 = performance.now();
    out.push(...(await embedder.embed(batch)));
    log("embed", `batch [${i}, ${Math.min(i + EMBED_BATCH_SIZE, texts.length)}) took ${(performance.now() - t0).toFixed(0)}ms`);
    onProgress?.(Math.min(i + EMBED_BATCH_SIZE, texts.length), texts.length);
    // Yield to the event loop between batches so the UI (spinner/progress
    // text) can actually repaint — embedding runs on the main thread.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  log("embed", `embedBatched done in ${(performance.now() - batchStart).toFixed(0)}ms`);
  return out;
}

export class LabChunkIndex {
  private chunks: Chunk[] = [];
  private embeddings: Float32Array[] = [];
  private bm25: Bm25Index | null = null;
  private embedder: EmbeddingProvider | null = null;

  async build(
    notes: Note[],
    chunker: ChunkingStrategy,
    embedder: EmbeddingProvider,
    onProgress?: (done: number, total: number) => void,
  ): Promise<void> {
    log("index", `build start: ${notes.length} notes, embedder model=${embedder.modelId}`);
    const buildStart = performance.now();
    const chunkStart = performance.now();
    const chunks: Chunk[] = [];
    for (const note of notes) {
      chunks.push(...chunker.chunk(note));
    }
    log("index", `chunking done: ${chunks.length} chunks in ${(performance.now() - chunkStart).toFixed(0)}ms`);
    this.chunks = chunks;
    this.embedder = embedder;
    this.embeddings =
      chunks.length > 0 ? await embedBatched(embedder, chunks.map((c) => c.text), onProgress) : [];
    const bm25Start = performance.now();
    this.bm25 = new Bm25Index(chunks.map((c) => ({ id: c.id, text: c.text })));
    log("index", `bm25 index built in ${(performance.now() - bm25Start).toFixed(0)}ms`);
    log("index", `build total: ${(performance.now() - buildStart).toFixed(0)}ms`);
  }

  async search(query: string, opts: { similarity: SimilarityMetric; fusion: FusionConfig; topK?: number }): Promise<LabSearchResult[]> {
    log("search", `search start: ${this.chunks.length} chunks, ${this.embeddings.length} embeddings, similarity=${opts.similarity}`);
    const searchStart = performance.now();
    const topK = opts.topK ?? 10;
    const scoreByChunkId = new Map<string, number>();

    if (opts.similarity === "bm25") {
      if (!this.bm25) throw new Error("Index not built.");
      log("search", "using bm25 similarity");
      const bm25Start = performance.now();
      const bm25Results = this.bm25.search(query, this.chunks.length || 1);
      log("search", `bm25 scored ${bm25Results.length} chunks in ${(performance.now() - bm25Start).toFixed(0)}ms`);
      for (const r of bm25Results) scoreByChunkId.set(r.id, r.score);
    } else {
      if (!this.embedder) throw new Error("Index not built.");
      log("search", `using embedding similarity (${opts.similarity}), embedding query...`);
      const embedStart = performance.now();
      const simFn = SIMILARITY_FNS[opts.similarity];
      const [queryEmbedding] = await this.embedder.embed([query]);
      log("search", `query embedding took ${(performance.now() - embedStart).toFixed(0)}ms`);
      if (queryEmbedding) {
        const scoreStart = performance.now();
        this.chunks.forEach((chunk, i) => {
          const emb = this.embeddings[i];
          if (!emb) return;
          scoreByChunkId.set(chunk.id, simFn(queryEmbedding, emb));
        });
        log("search", `scored ${this.chunks.length} chunks in ${(performance.now() - scoreStart).toFixed(0)}ms`);
      }
    }

    const chunkScores = this.chunks
      .map((c) => ({ noteId: c.noteId, layer: c.layer, score: scoreByChunkId.get(c.id) ?? 0 }))
      .filter((cs) => cs.score > 0);

    const fuseStart = performance.now();
    const noteScores = fuseChunkScores(chunkScores, opts.fusion);
    log("search", `fused ${chunkScores.length} chunk scores into ${noteScores.length} note scores in ${(performance.now() - fuseStart).toFixed(0)}ms`);

    const results = noteScores.slice(0, topK).map((ns) => {
      const noteChunks = this.chunks
        .filter((c) => c.noteId === ns.noteId)
        .map((c) => ({ chunk: c, score: scoreByChunkId.get(c.id) ?? 0 }))
        .filter((sc) => sc.score > 0)
        .sort((a, b) => b.score - a.score);
      return { noteId: ns.noteId, score: ns.score, matchedChunks: noteChunks };
    });
    log("search", `search total: ${(performance.now() - searchStart).toFixed(0)}ms, ${results.length}/${topK} top results`);
    return results;
  }
}
