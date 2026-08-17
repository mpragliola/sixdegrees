import { describe, expect, it } from "vitest";
import { cosineSimilarity } from "./similarity.js";

describe("similarity", () => {
  it("computes cosine similarity of identical vectors as 1", () => {
    const v = new Float32Array([1, 0, 0]);
    expect(cosineSimilarity(v, v)).toBeCloseTo(1);
  });

  it("computes cosine similarity of orthogonal vectors as 0", () => {
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([0, 1]))).toBeCloseTo(0);
  });

  it("computes cosine similarity of opposite vectors as -1", () => {
    expect(cosineSimilarity(new Float32Array([1, 2]), new Float32Array([-1, -2]))).toBeCloseTo(-1);
  });

  it("is scale-invariant for non-normalized inputs", () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([2, 4, 6]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(1);
  });

  it("returns 0 when a vector has zero norm", () => {
    expect(cosineSimilarity(new Float32Array([0, 0]), new Float32Array([1, 1]))).toBe(0);
  });
});
