import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import type { EmbeddingProvider, EmbeddingRole } from "./types.js";

/**
 * Storage-agnostic cache for computed (post-normalization) embedding
 * vectors, keyed by (modelId, text). No storage backend is implied here —
 * hosts (browser, Node, Electron) implement this against whatever they have
 * (filesystem, IndexedDB, in-memory, etc.) and pass an instance in via
 * TransformersEmbeddingProvider's constructor options.
 */
export interface EmbeddingCache {
  get(modelId: string, text: string): Promise<Float32Array | undefined>;
  set(modelId: string, text: string, embedding: Float32Array): Promise<void>;
}

/**
 * transformers.js backend selection note:
 *
 * `@huggingface/transformers` (the actively maintained successor to
 * `@xenova/transformers`) auto-detects Node vs. browser environments and
 * picks onnxruntime-node vs. onnxruntime-web via the package's own
 * conditional `exports` map (the "node" export condition resolves to
 * dist/transformers.node.mjs, everything else to dist/transformers.web.js).
 * This means bundlers and runtimes resolve the correct backend automatically
 * — no manual embedding.browser.ts / embedding.node.ts split is needed here.
 * A single implementation works across browser, Node, and Electron.
 */

const DEFAULT_MODEL_ID = "Xenova/all-MiniLM-L6-v2";

/**
 * Progress event shape forwarded from transformers.js's own
 * `progress_callback` during model download/initialization. Fields beyond
 * `status` are only present for the relevant status (e.g. `progress`/`loaded`/
 * `total` during "progress", `file` during "initiate"/"progress"/"done").
 */
export interface ModelLoadProgress {
  status: "initiate" | "download" | "progress" | "done" | "ready" | string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

/**
 * Per-model embedding configuration. Asymmetric retrieval models require
 * role prefixes (queries and passages embedded differently) and some are
 * CLS-pooled rather than mean-pooled; getting either wrong doesn't error —
 * it silently degrades retrieval quality, so the exact strings live here,
 * verified against each model's card.
 */
export interface EmbeddingModelConfig {
  dimensions?: number;
  /** Prepended to texts embedded with role "query". */
  queryPrefix?: string;
  /** Prepended to texts embedded with role "passage". */
  passagePrefix?: string;
  /** Pooling strategy; defaults to "mean". */
  pooling?: "mean" | "cls";
}

const BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";

/** Known configurations for common sentence-transformer models. */
const MODEL_CONFIGS: Record<string, EmbeddingModelConfig> = {
  "Xenova/all-MiniLM-L6-v2": { dimensions: 384 },
  "Xenova/all-mpnet-base-v2": { dimensions: 768 },
  "Xenova/gte-small": { dimensions: 384 },
  "Xenova/bge-small-en-v1.5": { dimensions: 384, pooling: "cls", queryPrefix: BGE_QUERY_PREFIX },
  "Snowflake/snowflake-arctic-embed-s": { dimensions: 384, pooling: "cls", queryPrefix: BGE_QUERY_PREFIX },
  "Xenova/multilingual-e5-small": { dimensions: 384, queryPrefix: "query: ", passagePrefix: "passage: " },
  "nomic-ai/nomic-embed-text-v1.5": {
    dimensions: 768,
    queryPrefix: "search_query: ",
    passagePrefix: "search_document: ",
  },
};

/** Resolve the config for a modelId; unknown models get symmetric mean-pooled defaults. */
export function getModelConfig(modelId: string): EmbeddingModelConfig {
  return MODEL_CONFIGS[modelId] ?? {};
}

function unitNormalize(vec: Float32Array): Float32Array {
  let normSq = 0;
  for (let i = 0; i < vec.length; i++) {
    normSq += vec[i]! * vec[i]!;
  }
  const norm = Math.sqrt(normSq);
  if (norm === 0) return vec;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i]! / norm;
  }
  return out;
}

/**
 * EmbeddingProvider backed by a local transformers.js feature-extraction
 * pipeline (in-process, no API keys). The underlying pipeline is cached per
 * modelId so repeated `embed()` calls (and repeated construction with the
 * same modelId) don't reload the model.
 *
 * Embeddings are unit-normalized at embed time, so cosine similarity
 * reduces to a plain dot product on the unit hypersphere.
 */
export class TransformersEmbeddingProvider implements EmbeddingProvider {
  readonly modelId: string;
  private _dimensions: number | null;
  private readonly config: EmbeddingModelConfig;
  private static pipelineCache = new Map<string, Promise<FeatureExtractionPipeline>>();

  private readonly onModelLoadProgress?: (progress: ModelLoadProgress) => void;
  private readonly cache?: EmbeddingCache;

  constructor(
    options: {
      modelId?: string;
      onModelLoadProgress?: (progress: ModelLoadProgress) => void;
      cache?: EmbeddingCache;
      /** Overrides the built-in MODEL_CONFIGS entry (or lack thereof) for this modelId. */
      modelConfig?: EmbeddingModelConfig;
    } = {},
  ) {
    this.modelId = options.modelId ?? DEFAULT_MODEL_ID;
    this.config = options.modelConfig ?? getModelConfig(this.modelId);
    this._dimensions = this.config.dimensions ?? null;
    this.onModelLoadProgress = options.onModelLoadProgress;
    this.cache = options.cache;
  }

  get dimensions(): number {
    if (this._dimensions === null) {
      throw new Error(
        `Dimensions for model "${this.modelId}" are not yet known — call embed() at least once first, ` +
          `or add it to MODEL_CONFIGS in embedding.ts.`,
      );
    }
    return this._dimensions;
  }

  private getPipeline(): Promise<FeatureExtractionPipeline> {
    let cached = TransformersEmbeddingProvider.pipelineCache.get(this.modelId);
    if (!cached) {
      cached = pipeline("feature-extraction", this.modelId, {
        progress_callback: this.onModelLoadProgress
          ? (p: ModelLoadProgress) => this.onModelLoadProgress!(p)
          : undefined,
      }) as Promise<FeatureExtractionPipeline>;
      TransformersEmbeddingProvider.pipelineCache.set(this.modelId, cached);
    }
    return cached;
  }

  /**
   * Trigger model download/init without embedding anything. Lets a host UI
   * drive an explicit "load this model" action (with progress via
   * onModelLoadProgress) instead of loading being an implicit side effect of
   * the first embed() call. Cached per modelId, so calling this and then
   * embed() does not load twice.
   */
  async load(): Promise<void> {
    await this.getPipeline();
  }

  /**
   * Clear the cached pipeline(s). With no modelId, clears all cached
   * pipelines; with a modelId, clears only that one. Subsequent embed()/load()
   * calls will re-trigger a fresh pipeline() call (and re-download if the
   * underlying transformers.js cache was also cleared).
   */
  static clearCache(modelId?: string): void {
    if (modelId) {
      TransformersEmbeddingProvider.pipelineCache.delete(modelId);
    } else {
      TransformersEmbeddingProvider.pipelineCache.clear();
    }
  }

  async embed(texts: string[], role: EmbeddingRole = "passage"): Promise<Float32Array[]> {
    if (texts.length === 0) return [];

    // Apply the model's role prefix before caching/embedding: prefixed text
    // IS the input from the model's point of view, so the (modelId, text)
    // cache keys off it and query/passage embeddings of the same string
    // never collide.
    const prefix = (role === "query" ? this.config.queryPrefix : this.config.passagePrefix) ?? "";
    if (prefix) texts = texts.map((t) => prefix + t);

    const results: (Float32Array | undefined)[] = new Array(texts.length);
    const missIndices: number[] = [];

    if (this.cache) {
      await Promise.all(
        texts.map(async (text, i) => {
          const cached = await this.cache!.get(this.modelId, text);
          if (cached) {
            results[i] = cached;
          } else {
            missIndices.push(i);
          }
        }),
      );
      missIndices.sort((a, b) => a - b);
    } else {
      for (let i = 0; i < texts.length; i++) missIndices.push(i);
    }

    if (missIndices.length > 0) {
      const extractor = await this.getPipeline();
      const missTexts = missIndices.map((i) => texts[i]!);
      const output = await extractor(missTexts, {
        pooling: this.config.pooling ?? "mean",
        normalize: false,
      });
      // output.dims: [batch, hiddenSize]; output.data: flat Float32Array-like.
      const dims = output.dims as number[];
      const hiddenSize = dims[dims.length - 1]!;
      const data = output.data as Float32Array;

      if (this._dimensions === null) {
        this._dimensions = hiddenSize;
      }

      for (let j = 0; j < missIndices.length; j++) {
        const i = missIndices[j]!;
        const slice = data.slice(j * hiddenSize, (j + 1) * hiddenSize);
        const vector = unitNormalize(slice);
        results[i] = vector;
        if (this.cache) await this.cache.set(this.modelId, texts[i]!, vector);
      }
    }

    return results as Float32Array[];
  }
}
