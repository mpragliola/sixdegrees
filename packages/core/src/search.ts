import type { Chunk, ChunkingStrategy, EmbeddingProvider, Note } from "./types.js";
import { cosineSimilarity, dotProduct, euclideanSimilarity, type SimilarityFn } from "./similarity.js";
import { fuseChunkScores, type ChunkScore, type FusionConfig } from "./fusion.js";
import { Bm25Index } from "./bm25.js";

export type SimilarityMetric = "cosine" | "dot" | "euclidean" | "bm25";

export interface SearchResult {
  noteId: string;
  score: number;
  matchedChunks: Chunk[];
}

export interface SearchOptions {
  similarity: SimilarityMetric;
  fusion: FusionConfig;
  topK?: number;
}

export interface SearchIndex {
  build(notes: Note[], chunker: ChunkingStrategy, embedder: EmbeddingProvider): Promise<void>;
  search(query: string, opts: SearchOptions): Promise<SearchResult[]>;
}

const SIMILARITY_FNS: Record<Exclude<SimilarityMetric, "bm25">, SimilarityFn> = {
  cosine: cosineSimilarity,
  dot: dotProduct,
  euclidean: euclideanSimilarity,
};

interface IndexedChunk {
  chunk: Chunk;
  embedding: Float32Array;
}

/**
 * In-memory search index: holds chunks + their embeddings as a flat array
 * (no vector DB — this is a lab/experimentation library). Also maintains a
 * Bm25Index over the same chunk texts so `similarity: "bm25"` can bypass
 * embeddings entirely while still going through the same fusion pipeline.
 *
 * The embedder passed to build() is retained so search() can embed the
 * query text with the same model used to build the index.
 */
export class InMemorySearchIndex implements SearchIndex {
  private indexed: IndexedChunk[] = [];
  private bm25: Bm25Index | null = null;
  private embedder: EmbeddingProvider | null = null;

  async build(notes: Note[], chunker: ChunkingStrategy, embedder: EmbeddingProvider): Promise<void> {
    const allChunks: Chunk[] = [];
    for (const note of notes) {
      allChunks.push(...chunker.chunk(note));
    }

    this.embedder = embedder;

    const texts = allChunks.map((c) => c.text);
    const embeddings = texts.length > 0 ? await embedder.embed(texts) : [];

    this.indexed = allChunks.map((chunk, i) => ({
      chunk,
      embedding: embeddings[i]!,
    }));

    this.bm25 = new Bm25Index(allChunks.map((c) => ({ id: c.id, text: c.text })));
  }

  async search(query: string, opts: SearchOptions): Promise<SearchResult[]> {
    const topK = opts.topK ?? 10;
    let chunkScores: ChunkScore[];
    let scoreByChunkId: Map<string, number>;

    if (opts.similarity === "bm25") {
      if (!this.bm25) {
        throw new Error("Index not built. Call build() before search().");
      }
      // Request scores for every chunk so fusion sees the full picture, not
      // just a pre-fusion top-K of raw chunks.
      const bm25Results = this.bm25.search(query, this.indexed.length || 1);
      scoreByChunkId = new Map(bm25Results.map((r) => [r.id, r.score]));
      chunkScores = this.indexed
        .map(({ chunk }) => ({
          noteId: chunk.noteId,
          layer: chunk.layer,
          score: scoreByChunkId.get(chunk.id) ?? 0,
        }))
        .filter((cs) => cs.score > 0);
    } else {
      if (!this.embedder) {
        throw new Error("Index not built. Call build() before search().");
      }
      const simFn = SIMILARITY_FNS[opts.similarity];
      const [queryEmbedding] = await this.embedder.embed([query]);
      if (!queryEmbedding) {
        chunkScores = [];
        scoreByChunkId = new Map();
      } else {
        scoreByChunkId = new Map();
        chunkScores = this.indexed.map(({ chunk, embedding }) => {
          const score = simFn(queryEmbedding, embedding);
          scoreByChunkId.set(chunk.id, score);
          return { noteId: chunk.noteId, layer: chunk.layer, score };
        });
      }
    }

    const noteScores = fuseChunkScores(chunkScores, opts.fusion);

    return noteScores.slice(0, topK).map((ns) => ({
      noteId: ns.noteId,
      score: ns.score,
      matchedChunks: this.indexed
        .filter((ic) => ic.chunk.noteId === ns.noteId)
        .map((ic) => ic.chunk)
        .sort((a, b) => (scoreByChunkId.get(b.id) ?? 0) - (scoreByChunkId.get(a.id) ?? 0)),
    }));
  }
}
