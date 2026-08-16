# sixdegrees

Chunking + embedding + similarity + fusion search library for notetaker semantic search. In-process, no vector DB, no API keys — embeddings run locally via `@huggingface/transformers`. Works in browser, Node, and Electron.

## Install

```bash
pnpm add sixdegrees
```

Within this workspace, the `lab` package already depends on it via `workspace:*`.

## Core concepts

A search pipeline is assembled from four independent pieces:

1. **Chunking** — split a `Note` into `Chunk[]` (`ChunkingStrategy`)
2. **Embedding** — turn chunk text into vectors (`EmbeddingProvider`)
3. **Similarity** — score a query vector against chunk vectors, or use lexical TF-IDF instead
4. **Fusion** — combine per-chunk scores into per-note scores

`InMemorySearchIndex` wires all four together.

## Quick start

```ts
import {
  InMemorySearchIndex,
  TitleAwareLayeredChunker,
  TransformersEmbeddingProvider,
} from "sixdegrees";

const notes = [
  { id: "1", title: "Sourdough notes", body: "Hydration around 75%.\n\nBulk ferment 4-6 hours at room temp." },
  { id: "2", title: "Trip planning", body: "Book flights first, then lodging." },
];

const chunker = new TitleAwareLayeredChunker();
const embedder = new TransformersEmbeddingProvider(); // Xenova/all-MiniLM-L6-v2 by default

const index = new InMemorySearchIndex();
await index.build(notes, chunker, embedder);

const results = await index.search("bread hydration", {
  similarity: "cosine",
  fusion: { method: "weighted-sum", layerWeights: { title: 2, paragraph: 1, note: 1 } },
  topK: 5,
});

for (const r of results) {
  console.log(r.noteId, r.score, r.matchedChunks.map((c) => c.layer));
}
```

## Types (`types.ts`)

```ts
interface Note {
  id: string;
  title: string;
  body: string;
  createdAt?: string;
}

interface Chunk {
  id: string;
  noteId: string;
  layer: string;       // e.g. "title", "note", "paragraph", "sentence-window"
  text: string;
  charStart: number;    // offset into the source text (see each chunker's docs below)
  charEnd: number;
}

interface ChunkingStrategy {
  readonly name: string;
  chunk(note: { id: string; title: string; body: string }): Chunk[];
}

interface EmbeddingProvider {
  readonly modelId: string;
  readonly dimensions: number;
  load?(): Promise<void>;             // optional explicit warm-up
  embed(texts: string[]): Promise<Float32Array[]>;
}
```

## Chunking strategies (`chunking/*`)

All chunkers implement `ChunkingStrategy` and produce `Chunk[]`. Empty title/body input yields `[]`.

| Class | `name` | `layer` | Notes |
|---|---|---|---|
| `TitleOnlyChunker` | `title-only` | `title` | One chunk = trimmed title. Offsets index into `note.title`. |
| `WholeNoteChunker` | `whole-note` | `note` | One chunk = `title + "\n\n" + body`, trimmed. |
| `ParagraphChunker` | `paragraph` | `paragraph` | Splits `note.body` on blank-line runs. Offsets index into `note.body`. |
| `SentenceWindowChunker` | `sentence-window` | `sentence-window` | Sliding window over naively-split sentences (see below). |
| `LayeredChunker` | `layered` (custom) | (per sub-strategy) | Runs a list of strategies and concatenates their output. |
| `TitleAwareLayeredChunker` | `title-aware-layered` | (per sub-strategy) | Preset `LayeredChunker` of `[TitleOnlyChunker, WholeNoteChunker, ParagraphChunker]`. |

### `SentenceWindowChunker`

```ts
new SentenceWindowChunker({ windowSize?: number; overlap?: number })
```

- `windowSize` (default `3`) — sentences per window; must be `> 0`.
- `overlap` (default `1`) — sentence overlap between consecutive windows; must be `>= 0` and `< windowSize`.
- Sentence splitting is a naive regex on `.`/`!`/`?` followed by whitespace — it does **not** guard against abbreviations (e.g. "Dr. Smith" splits incorrectly). Deliberate simplification, not a bug.

### `LayeredChunker`

```ts
new LayeredChunker(strategies: ChunkingStrategy[], name = "layered")
```

Runs every strategy in `strategies` against the note and concatenates their chunks (each sub-strategy keeps its own `layer` tag). Use this to build custom layer combinations; `TitleAwareLayeredChunker` is a ready-made preset.

## Embedding (`embedding.ts`)

### `TransformersEmbeddingProvider`

```ts
new TransformersEmbeddingProvider(options?: {
  modelId?: string;                                   // default "Xenova/all-MiniLM-L6-v2"
  onModelLoadProgress?: (progress: ModelLoadProgress) => void;
  cache?: EmbeddingCache;
})
```

- Backed by a local `@huggingface/transformers` feature-extraction pipeline — no API key, runs in-process. The library auto-selects onnxruntime-node vs. onnxruntime-web, so the same code runs in Node, browser, and Electron.
- The underlying pipeline is cached **per `modelId`** (static cache), so repeated construction/`embed()` calls with the same model don't reload it.
- Vectors are **unit-normalized** at embed time, which is why cosine, dot product, and euclidean similarity are all directly comparable/interchangeable downstream (see Similarity).
- `dimensions` is known up front for `Xenova/all-MiniLM-L6-v2` (384) and `Xenova/all-mpnet-base-v2` (768); for any other model it throws until `embed()` has been called at least once.

Methods:

- `load(): Promise<void>` — explicitly trigger model download/init (e.g. to drive a loading UI via `onModelLoadProgress`). Optional; `embed()` will lazily load on first use if you skip it. Safe to call repeatedly — dedupes.
- `embed(texts: string[]): Promise<Float32Array[]>` — embeds a batch. Empty input returns `[]`. If an `EmbeddingCache` was passed in, per-text hits are served from cache and only misses go through the pipeline (then written back to cache).
- `static clearCache(modelId?: string): void` — evicts the static pipeline cache; no `modelId` clears everything. Next `embed()`/`load()` reloads (and re-downloads if the transformers.js-level cache was also cleared).

### `EmbeddingCache`

```ts
interface EmbeddingCache {
  get(modelId: string, text: string): Promise<Float32Array | undefined>;
  set(modelId: string, text: string, embedding: Float32Array): Promise<void>;
}
```

Storage-agnostic — no backend is implied here. Pass in whatever fits the host (filesystem, IndexedDB, in-memory Map, etc.) via the `cache` constructor option. The `lab` package ships an IndexedDB-backed implementation as a reference.

### `ModelLoadProgress`

```ts
interface ModelLoadProgress {
  status: "initiate" | "download" | "progress" | "done" | "ready" | string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}
```

Forwarded as-is from transformers.js's own `progress_callback`. Fields beyond `status` are only populated for the relevant status.

## Similarity (`similarity.ts`)

All three operate on `Float32Array` vectors and assume unit-normalized input (guaranteed if produced by `TransformersEmbeddingProvider`), so they stay comparable in scale and "higher is always better":

```ts
type SimilarityFn = (a: Float32Array, b: Float32Array) => number;

dotProduct(a, b): number
cosineSimilarity(a, b): number      // safe on non-normalized input too (divides by norms); returns 0 if either norm is 0
euclideanSimilarity(a, b): number   // 1 / (1 + euclideanDistance(a, b)) — distance range [0, 2] on unit vectors
```

## Lexical search (`tfidf.ts`)

### `TfIdfIndex`

```ts
new TfIdfIndex(docs: { id: string; text: string }[])
```

Standalone TF-IDF index — independent of the embedding pathway. Tokenizes on Unicode letters/numbers (lowercased, punctuation stripped), uses smoothed IDF (`log(N/df) + 1`), and scores cosine similarity between TF-IDF vectors.

```ts
index.search(query: string, topK = 10): { id: string; score: number }[]
```

Returns up to `topK` results sorted by descending score. Wired into search via `SearchOptions.similarity = "tfidf"`, which bypasses embeddings entirely but still flows through the same fusion pipeline as embedding-based search.

## Fusion (`fusion.ts`)

Combines per-chunk scores into per-note scores across chunking layers.

```ts
type FusionMethod = "max" | "weighted-sum";

interface FusionConfig {
  method: FusionMethod;
  layerWeights?: Record<string, number>; // "weighted-sum" only; missing layers default to weight 1
}

fuseChunkScores(chunkScores: ChunkScore[], config: FusionConfig): NoteScore[]
```

- **`"max"`** — note score = max score across all its chunks. Layer weights are ignored.
- **`"weighted-sum"`** (the intended default fusion strategy) — for each note, take the best score per layer present, multiply by that layer's weight (default `1`), and sum across layers.

Reciprocal Rank Fusion (RRF) was considered and deliberately **not** implemented.

## Search orchestration (`search.ts`)

### `InMemorySearchIndex`

In-memory index: chunks + embeddings held as a flat array (no vector DB — this is a lab/experimentation library) plus a `TfIdfIndex` built over the same chunk texts so lexical search stays available without re-embedding.

```ts
class InMemorySearchIndex implements SearchIndex {
  build(notes: Note[], chunker: ChunkingStrategy, embedder: EmbeddingProvider): Promise<void>;
  search(query: string, opts: SearchOptions): Promise<SearchResult[]>;
}
```

```ts
type SimilarityMetric = "cosine" | "dot" | "euclidean" | "tfidf";

interface SearchOptions {
  similarity: SimilarityMetric;
  fusion: FusionConfig;
  topK?: number; // default 10
}

interface SearchResult {
  noteId: string;
  score: number;
  matchedChunks: Chunk[]; // chunks belonging to this note, sorted by descending chunk score
}
```

- `build()` chunks every note, embeds all chunk texts in one batch via `embedder.embed()`, and builds the TF-IDF index over the same chunks. The `embedder` instance is retained so `search()` later embeds the query with the same model.
- `search()` with `similarity: "tfidf"` scores every indexed chunk via `TfIdfIndex.search()` (requesting scores for the full chunk set, not a pre-fusion top-K) and drops zero-score chunks before fusion.
- `search()` with `"cosine" | "dot" | "euclidean"` embeds the query once and scores it against every indexed chunk embedding with the corresponding `SimilarityFn`.
- Either path then calls `fuseChunkScores()` with `opts.fusion`, slices to `topK`, and attaches each result's `matchedChunks` (chunks for that note, sorted by their own descending score).
- Calling `search()` before `build()` throws.

## Notes on cross-cutting design

- Every score across the library ("higher is better") is enforced consistently: euclidean distance is inverted to a similarity, TF-IDF and embedding paths both produce ascending-is-worse / descending-is-better scores, and fusion sorts descending.
- `EmbeddingProvider` and `EmbeddingCache` are both intentionally environment-agnostic — no browser/Node globals leak into the interfaces. Host apps supply the concrete cache backend.
- The library has no notion of persistence beyond the optional `EmbeddingCache` — `InMemorySearchIndex` is rebuilt from `Note[]` on every `build()` call.
