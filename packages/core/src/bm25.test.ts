import { describe, expect, it } from "vitest";
import { Bm25Index } from "./bm25.js";

function score(results: { id: string; score: number }[], id: string): number {
  return results.find((r) => r.id === id)?.score ?? 0;
}

describe("Bm25Index", () => {
  it("returns matching docs ranked above non-matching ones", () => {
    const index = new Bm25Index([
      { id: "a", text: "the cat sat on the mat" },
      { id: "b", text: "dogs chase balls in the park" },
    ]);
    const results = index.search("cat");
    expect(results[0]!.id).toBe("a");
    expect(results[0]!.score).toBeGreaterThan(0);
    expect(score(results, "b")).toBe(0);
  });

  it("returns [] for an empty query", () => {
    const index = new Bm25Index([{ id: "a", text: "hello world" }]);
    expect(index.search("")).toEqual([]);
    expect(index.search("!!!")).toEqual([]);
  });

  it("respects topK", () => {
    const index = new Bm25Index([
      { id: "a", text: "cat" },
      { id: "b", text: "cat cat" },
      { id: "c", text: "cat cat cat" },
    ]);
    expect(index.search("cat", 2)).toHaveLength(2);
  });

  it("saturates term frequency: repeated-term gains diminish", () => {
    // Same doc length everywhere (padded with unique filler) so only tf varies.
    const index = new Bm25Index([
      { id: "tf1", text: "cat x1 x2 x3 x4 x5 x6 x7" },
      { id: "tf2", text: "cat cat x1 x2 x3 x4 x5 x6" },
      { id: "tf4", text: "cat cat cat cat x1 x2 x3 x4" },
    ]);
    const results = index.search("cat", 10);
    const s1 = score(results, "tf1");
    const s2 = score(results, "tf2");
    const s4 = score(results, "tf4");
    expect(s2).toBeGreaterThan(s1);
    expect(s4).toBeGreaterThan(s2);
    // Diminishing returns: going 2 -> 4 occurrences gains less than 1 -> 2.
    expect(s4 - s2).toBeLessThan(s2 - s1);
    // And the score is bounded by idf * (k1 + 1).
    const bound = (Math.log(1 + (3 - 3 + 0.5) / (3 + 0.5))) * (1.2 + 1);
    expect(s4).toBeLessThan(bound + 1e-9);
  });

  it("normalizes by length: shorter doc with same tf scores higher", () => {
    const index = new Bm25Index([
      { id: "short", text: "cat runs" },
      { id: "long", text: "cat runs and jumps and sleeps all day long in the sunny garden" },
    ]);
    const results = index.search("cat");
    expect(score(results, "short")).toBeGreaterThan(score(results, "long"));
  });

  it("ignores length when b = 0", () => {
    const index = new Bm25Index(
      [
        { id: "short", text: "cat runs" },
        { id: "long", text: "cat runs and jumps and sleeps all day long" },
      ],
      { b: 0 },
    );
    const results = index.search("cat");
    expect(score(results, "short")).toBeCloseTo(score(results, "long"));
  });

  it("lets rare-term idf dominate common terms", () => {
    // "common" appears in every doc; "zyzzyva" only in one.
    const index = new Bm25Index([
      { id: "rare", text: "common zyzzyva here" },
      { id: "c1", text: "common words only" },
      { id: "c2", text: "common stuff again" },
      { id: "c3", text: "common filler text" },
    ]);
    const results = index.search("common zyzzyva");
    // Doc with the rare term wins, and by a wide margin over common-only docs.
    expect(results[0]!.id).toBe("rare");
    expect(score(results, "rare")).toBeGreaterThan(2 * score(results, "c1"));
  });

  it("computes idf as ln(1 + (N - df + 0.5) / (df + 0.5))", () => {
    // Single doc, single term, doc length == avg length, tf = 1:
    // lengthNorm = 1, tf part = (1 * (k1+1)) / (1 + k1) = 1 => score = idf.
    const index = new Bm25Index([{ id: "a", text: "cat" }]);
    const expectedIdf = Math.log(1 + (1 - 1 + 0.5) / (1 + 0.5));
    expect(index.search("cat")[0]!.score).toBeCloseTo(expectedIdf, 10);
  });

  it("weights repeated query terms by their count", () => {
    const index = new Bm25Index([
      { id: "cat", text: "cat filler words here" },
      { id: "dog", text: "dog filler words here" },
    ]);
    const single = index.search("cat dog");
    const doubled = index.search("cat cat dog");
    // With "cat" doubled in the query, the cat doc pulls ahead.
    expect(score(single, "cat")).toBeCloseTo(score(single, "dog"));
    expect(score(doubled, "cat")).toBeGreaterThan(score(doubled, "dog"));
    expect(score(doubled, "cat")).toBeCloseTo(2 * score(single, "cat"));
  });

  it("accepts custom k1: k1 = 0 removes tf sensitivity", () => {
    const index = new Bm25Index(
      [
        { id: "tf1", text: "cat x1 x2 x3" },
        { id: "tf3", text: "cat cat cat x1" },
      ],
      { k1: 0 },
    );
    const results = index.search("cat");
    expect(score(results, "tf1")).toBeCloseTo(score(results, "tf3"));
  });

  it("handles an empty corpus", () => {
    const index = new Bm25Index([]);
    expect(index.search("anything")).toEqual([]);
  });
});
