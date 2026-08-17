/**
 * Vector similarity. Embeddings are unit-normalized at embed time (see
 * embedding.ts), so cosine similarity is the single vector metric — on unit
 * vectors, dot product equals cosine and euclidean distance is a monotone
 * transform of it, so all three produce identical rankings.
 */

export type SimilarityFn = (a: Float32Array, b: Float32Array) => number;

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return sum;
}

/**
 * Cosine similarity. If inputs are already unit-normalized (as produced by
 * TransformersEmbeddingProvider), this reduces to a dot product — the norm
 * division below is kept so this function is correct even when given
 * non-normalized vectors.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  const d = dot(a, b);
  const normA = Math.sqrt(dot(a, a));
  const normB = Math.sqrt(dot(b, b));
  if (normA === 0 || normB === 0) return 0;
  return d / (normA * normB);
}
