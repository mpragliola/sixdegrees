/**
 * Shared types for sixdegrees: notes, chunks, embedding providers.
 */

export interface Note {
  id: string;
  title: string;
  body: string;
  createdAt?: string;
}

export interface Chunk {
  id: string;
  noteId: string;
  /** e.g. "title", "note", "paragraph", "sentence-window" */
  layer: string;
  text: string;
  charStart: number;
  charEnd: number;
  /**
   * Optional alternate text to use when embedding/indexing this chunk (e.g.
   * the note title prepended for context: `${title}\n\n${text}`). When
   * absent, `text` is embedded as-is. `text`/charStart/charEnd always keep
   * describing the original span in the note body, so consumers that
   * highlight or display the chunk should keep using `text`.
   */
  embeddingText?: string;
}

export interface ChunkingStrategy {
  readonly name: string;
  chunk(note: { id: string; title: string; body: string }): Chunk[];
}

/**
 * Whether a text is a search query or an indexed passage. Asymmetric
 * retrieval models (bge, arctic, e5, nomic, …) are trained with distinct
 * query/passage prefixes and silently degrade if both sides are embedded
 * identically; symmetric models ignore the distinction.
 */
export type EmbeddingRole = "query" | "passage";

/**
 * Embedding provider abstraction. Implementations must work across browser,
 * Node, and Electron without leaking environment-specific globals into the
 * shared interface.
 */
export interface EmbeddingProvider {
  readonly modelId: string;
  readonly dimensions: number;
  /**
   * Explicitly trigger model loading (download/init). Optional: callers that
   * don't need up-front progress reporting can skip this and let embed()
   * load lazily on first use. Safe to call multiple times — implementations
   * should dedupe concurrent/repeat calls.
   */
  load?(): Promise<void>;
  /** `role` defaults to "passage"; pass "query" when embedding search queries. */
  embed(texts: string[], role?: EmbeddingRole): Promise<Float32Array[]>;
}
