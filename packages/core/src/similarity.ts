/**
 * Similarity functions. All three are designed to operate on unit-normalized
 * vectors (see embedding.ts, which normalizes at embed time) so that cosine,
 * dot product, and euclidean distance are directly comparable in scale and
 * "higher is always better" — euclidean distance is converted to a
 * similarity via `1 / (1 + distance)`.
 */

export type SimilarityFn = (a: Float32Array, b: Float32Array) => number;

export function dotProduct(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return sum;
}

/**
 * Cosine similarity. If inputs are already unit-normalized (as produced by
 * TransformersEmbeddingProvider), this is equivalent to dotProduct — the
 * norm division below is kept so this function is correct even when given
 * non-normalized vectors.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  const dot = dotProduct(a, b);
  const normA = Math.sqrt(dotProduct(a, a));
  const normB = Math.sqrt(dotProduct(b, b));
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}

function euclideanDistance(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Euclidean similarity, mapped from distance so that higher is always
 * better, consistent with cosine/dot: `1 / (1 + distance)`.
 * Still meaningful on normalized vectors (distance range becomes [0, 2]).
 */
export function euclideanSimilarity(a: Float32Array, b: Float32Array): number {
  return 1 / (1 + euclideanDistance(a, b));
}
