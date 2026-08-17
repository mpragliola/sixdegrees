/**
 * Score fusion: combines per-chunk similarity/tfidf scores into a single
 * per-note score across chunking layers.
 *
 * "max": note score = max over all its chunk scores, layer weights ignored.
 *
 * "weighted-sum": for each note, take the BEST chunk score per layer that is
 * present for that note, multiply by that layer's weight (default 1 if
 * unspecified in layerWeights), sum across layers, then divide by the total
 * weight of the layers PRESENT for that note. The normalization keeps notes
 * comparable when they lack some layers (e.g. a body-less note has no
 * paragraph chunks and would otherwise be structurally capped below fuller
 * notes). This is the default fusion strategy per user choice — RRF was
 * explicitly rejected and is not implemented.
 */

export type FusionMethod = "max" | "weighted-sum";

export interface FusionConfig {
  method: FusionMethod;
  /** Used only when method === "weighted-sum". Missing layers default to weight 1. */
  layerWeights?: Record<string, number>;
}

export interface ChunkScore {
  noteId: string;
  layer: string;
  score: number;
}

export interface NoteScore {
  noteId: string;
  score: number;
}

export function fuseChunkScores(chunkScores: ChunkScore[], config: FusionConfig): NoteScore[] {
  if (config.method === "max") {
    const best = new Map<string, number>();
    for (const cs of chunkScores) {
      const prev = best.get(cs.noteId);
      if (prev === undefined || cs.score > prev) {
        best.set(cs.noteId, cs.score);
      }
    }
    return Array.from(best.entries())
      .map(([noteId, score]) => ({ noteId, score }))
      .sort((a, b) => b.score - a.score);
  }

  // weighted-sum: best score per (noteId, layer), weighted, then summed per noteId.
  const bestPerLayer = new Map<string, Map<string, number>>();
  for (const cs of chunkScores) {
    let layerMap = bestPerLayer.get(cs.noteId);
    if (!layerMap) {
      layerMap = new Map();
      bestPerLayer.set(cs.noteId, layerMap);
    }
    const prev = layerMap.get(cs.layer);
    if (prev === undefined || cs.score > prev) {
      layerMap.set(cs.layer, cs.score);
    }
  }

  const results: NoteScore[] = [];
  for (const [noteId, layerMap] of bestPerLayer.entries()) {
    let sum = 0;
    let weightSum = 0;
    for (const [layer, score] of layerMap.entries()) {
      const weight = config.layerWeights?.[layer] ?? 1;
      sum += score * weight;
      weightSum += weight;
    }
    results.push({ noteId, score: weightSum > 0 ? sum / weightSum : 0 });
  }
  results.sort((a, b) => b.score - a.score);
  return results;
}
