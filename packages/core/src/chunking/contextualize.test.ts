import { describe, expect, it } from "vitest";
import { ParagraphChunker } from "./paragraph.js";
import { SentenceWindowChunker } from "./sentenceWindow.js";
import { TitleAwareLayeredChunker } from "./layered.js";

const note = {
  id: "n1",
  title: "My Note",
  body: "First paragraph here.\n\nSecond paragraph. It has two sentences.",
};

describe("ParagraphChunker contextualize", () => {
  it("does not set embeddingText by default", () => {
    const chunks = new ParagraphChunker().chunk(note);
    expect(chunks.length).toBe(2);
    for (const c of chunks) expect(c.embeddingText).toBeUndefined();
  });

  it("does not set embeddingText when contextualize is false", () => {
    const chunks = new ParagraphChunker({ contextualize: false }).chunk(note);
    for (const c of chunks) expect(c.embeddingText).toBeUndefined();
  });

  it("prepends title in embeddingText when contextualize is true", () => {
    const chunks = new ParagraphChunker({ contextualize: true }).chunk(note);
    expect(chunks.length).toBe(2);
    expect(chunks[0]!.embeddingText).toBe("My Note\n\nFirst paragraph here.");
    expect(chunks[1]!.embeddingText).toBe("My Note\n\nSecond paragraph. It has two sentences.");
  });

  it("leaves text and offsets unchanged when contextualized", () => {
    const plain = new ParagraphChunker().chunk(note);
    const ctx = new ParagraphChunker({ contextualize: true }).chunk(note);
    expect(ctx.map((c) => ({ text: c.text, charStart: c.charStart, charEnd: c.charEnd }))).toEqual(
      plain.map((c) => ({ text: c.text, charStart: c.charStart, charEnd: c.charEnd })),
    );
    // Offsets still point into the original body.
    for (const c of ctx) {
      expect(note.body.slice(c.charStart, c.charEnd)).toBe(c.text);
    }
  });

  it("skips embeddingText when title is empty or whitespace", () => {
    for (const title of ["", "   "]) {
      const chunks = new ParagraphChunker({ contextualize: true }).chunk({ ...note, title });
      expect(chunks.length).toBe(2);
      for (const c of chunks) expect(c.embeddingText).toBeUndefined();
    }
  });
});

describe("SentenceWindowChunker contextualize", () => {
  const swNote = {
    id: "n2",
    title: "Windows",
    body: "One. Two. Three. Four. Five.",
  };

  it("does not set embeddingText by default", () => {
    const chunks = new SentenceWindowChunker({ windowSize: 2, overlap: 0 }).chunk(swNote);
    expect(chunks.length).toBeGreaterThan(0);
    for (const c of chunks) expect(c.embeddingText).toBeUndefined();
  });

  it("prepends title in embeddingText when contextualize is true", () => {
    const chunks = new SentenceWindowChunker({ windowSize: 2, overlap: 0, contextualize: true }).chunk(swNote);
    expect(chunks.length).toBeGreaterThan(0);
    for (const c of chunks) {
      expect(c.embeddingText).toBe(`Windows\n\n${c.text}`);
    }
  });

  it("leaves text and offsets unchanged when contextualized", () => {
    const plain = new SentenceWindowChunker({ windowSize: 2, overlap: 1 }).chunk(swNote);
    const ctx = new SentenceWindowChunker({ windowSize: 2, overlap: 1, contextualize: true }).chunk(swNote);
    expect(ctx.map((c) => ({ text: c.text, charStart: c.charStart, charEnd: c.charEnd }))).toEqual(
      plain.map((c) => ({ text: c.text, charStart: c.charStart, charEnd: c.charEnd })),
    );
    for (const c of ctx) {
      expect(swNote.body.slice(c.charStart, c.charEnd).trim()).toBe(c.text);
    }
  });

  it("skips embeddingText when title is empty", () => {
    const chunks = new SentenceWindowChunker({ contextualize: true }).chunk({ ...swNote, title: "" });
    for (const c of chunks) expect(c.embeddingText).toBeUndefined();
  });
});

describe("TitleAwareLayeredChunker contextualize", () => {
  it("contextualizes only the paragraph layer", () => {
    const chunks = new TitleAwareLayeredChunker({ contextualize: true }).chunk(note);
    const byLayer = (layer: string) => chunks.filter((c) => c.layer === layer);
    expect(byLayer("title").length).toBe(1);
    expect(byLayer("note").length).toBe(1);
    expect(byLayer("paragraph").length).toBe(2);

    for (const c of byLayer("title")) expect(c.embeddingText).toBeUndefined();
    for (const c of byLayer("note")) expect(c.embeddingText).toBeUndefined();
    for (const c of byLayer("paragraph")) {
      expect(c.embeddingText).toBe(`My Note\n\n${c.text}`);
    }
  });

  it("sets no embeddingText anywhere by default", () => {
    const chunks = new TitleAwareLayeredChunker().chunk(note);
    for (const c of chunks) expect(c.embeddingText).toBeUndefined();
  });
});
