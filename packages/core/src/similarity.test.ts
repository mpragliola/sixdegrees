import { describe, expect, it } from "vitest";
import { cosineSimilarity, dotProduct } from "./similarity.js";

describe("similarity", () => {
  it("computes dot product", () => {
    expect(dotProduct(new Float32Array([1, 2, 3]), new Float32Array([4, 5, 6]))).toBe(32);
  });

  it("computes cosine similarity of identical vectors as 1", () => {
    const v = new Float32Array([1, 0, 0]);
    expect(cosineSimilarity(v, v)).toBeCloseTo(1);
  });
});
