import { describe, expect, it } from "vitest";
import { fuseChunkScores } from "./fusion.js";

describe("fuseChunkScores", () => {
  it("max: takes the best chunk score per note", () => {
    const result = fuseChunkScores(
      [
        { noteId: "a", layer: "note", score: 0.4 },
        { noteId: "a", layer: "paragraph", score: 0.9 },
        { noteId: "b", layer: "note", score: 0.6 },
      ],
      { method: "max" },
    );
    expect(result).toEqual([
      { noteId: "a", score: 0.9 },
      { noteId: "b", score: 0.6 },
    ]);
  });

  it("weighted-sum: uses the best score per layer, weighted and normalized", () => {
    const result = fuseChunkScores(
      [
        { noteId: "a", layer: "title", score: 0.5 },
        { noteId: "a", layer: "paragraph", score: 0.8 },
        { noteId: "a", layer: "paragraph", score: 0.2 },
      ],
      { method: "weighted-sum", layerWeights: { title: 2, paragraph: 1 } },
    );
    // (0.5*2 + 0.8*1) / (2 + 1)
    expect(result[0]!.score).toBeCloseTo(1.8 / 3);
  });

  it("weighted-sum: does not penalize notes missing a layer", () => {
    // Both notes score 0.7 on every layer they have; the body-less note
    // (title only) must not be capped below the fuller note.
    const result = fuseChunkScores(
      [
        { noteId: "full", layer: "title", score: 0.7 },
        { noteId: "full", layer: "note", score: 0.7 },
        { noteId: "full", layer: "paragraph", score: 0.7 },
        { noteId: "title-only", layer: "title", score: 0.7 },
      ],
      { method: "weighted-sum" },
    );
    const scores = new Map(result.map((r) => [r.noteId, r.score]));
    expect(scores.get("title-only")!).toBeCloseTo(scores.get("full")!);
  });

  it("weighted-sum: a zero-weight layer is excluded from the normalization", () => {
    const result = fuseChunkScores(
      [
        { noteId: "a", layer: "title", score: 0.1 },
        { noteId: "a", layer: "paragraph", score: 0.9 },
      ],
      { method: "weighted-sum", layerWeights: { title: 0, paragraph: 1 } },
    );
    expect(result[0]!.score).toBeCloseTo(0.9);
  });

  it("weighted-sum: all layers weighted zero yields score 0", () => {
    const result = fuseChunkScores([{ noteId: "a", layer: "title", score: 0.9 }], {
      method: "weighted-sum",
      layerWeights: { title: 0 },
    });
    expect(result[0]!.score).toBe(0);
  });
});
