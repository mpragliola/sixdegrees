import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import type { EmbeddingProvider } from "./types.js";

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

/** Known embedding dimensions for common sentence-transformer models. */
const KNOWN_DIMENSIONS: Record<string, number> = {
  "Xenova/all-MiniLM-L6-v2": 384,
  "Xenova/all-mpnet-base-v2": 768,
};

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
  private static pipelineCache = new Map<string, Promise<FeatureExtractionPipeline>>();

  private readonly onModelLoadProgress?: (progress: ModelLoadProgress) => void;
  private readonly cache?: EmbeddingCache;

  constructor(
    options: {
      modelId?: string;
      onModelLoadProgress?: (progress: ModelLoadProgress) => void;
      cache?: EmbeddingCache;
    } = {},
  ) {
    this.modelId = options.modelId ?? DEFAULT_MODEL_ID;
    this._dimensions = KNOWN_DIMENSIONS[this.modelId] ?? null;
    this.onModelLoadProgress = options.onModelLoadProgress;
    this.cache = options.cache;
  }

  get dimensions(): number {
    if (this._dimensions === null) {
      throw new Error(
        `Dimensions for model "${this.modelId}" are not yet known — call embed() at least once first, ` +
          `or add it to KNOWN_DIMENSIONS in embedding.ts.`,
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

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];

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
      const output = await extractor(missTexts, { pooling: "mean", normalize: false });
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
