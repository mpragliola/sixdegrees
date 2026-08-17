import { describe, expect, it } from "vitest";
import { getModelConfig } from "./embedding.js";

describe("getModelConfig", () => {
  it("returns symmetric mean-pooled defaults for unknown models", () => {
    const config = getModelConfig("someone/some-model");
    expect(config.queryPrefix).toBeUndefined();
    expect(config.passagePrefix).toBeUndefined();
    expect(config.pooling).toBeUndefined();
  });

  it("keeps the current defaults symmetric drop-ins", () => {
    for (const id of ["Xenova/all-MiniLM-L6-v2", "Xenova/all-mpnet-base-v2", "Xenova/gte-small"]) {
      const config = getModelConfig(id);
      expect(config.queryPrefix).toBeUndefined();
      expect(config.passagePrefix).toBeUndefined();
      expect(config.pooling ?? "mean").toBe("mean");
    }
  });

  it("configures bge/arctic as CLS-pooled with the bge query prefix", () => {
    for (const id of ["Xenova/bge-small-en-v1.5", "Snowflake/snowflake-arctic-embed-s"]) {
      const config = getModelConfig(id);
      expect(config.pooling).toBe("cls");
      expect(config.queryPrefix).toBe("Represent this sentence for searching relevant passages: ");
      expect(config.passagePrefix).toBeUndefined();
    }
  });

  it("configures e5 and nomic with both role prefixes", () => {
    expect(getModelConfig("Xenova/multilingual-e5-small")).toMatchObject({
      queryPrefix: "query: ",
      passagePrefix: "passage: ",
    });
    expect(getModelConfig("nomic-ai/nomic-embed-text-v1.5")).toMatchObject({
      queryPrefix: "search_query: ",
      passagePrefix: "search_document: ",
    });
  });
});
