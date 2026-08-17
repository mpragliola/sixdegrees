import type { Chunk, ChunkingStrategy } from "../types.js";
import { TitleOnlyChunker } from "./titleOnly.js";
import { WholeNoteChunker } from "./wholeNote.js";
import { ParagraphChunker } from "./paragraph.js";

/**
 * Composes multiple ChunkingStrategy instances, running all of them against
 * a note and concatenating their output chunks (each sub-strategy already
 * tags its own chunks with its own `layer`).
 */
export class LayeredChunker implements ChunkingStrategy {
  constructor(private readonly strategies: ChunkingStrategy[], readonly name = "layered") {
    this.strategies = strategies;
    this.name = name;
  }

  chunk(note: { id: string; title: string; body: string }): Chunk[] {
    const chunks: Chunk[] = [];
    for (const strategy of this.strategies) {
      chunks.push(...strategy.chunk(note));
    }
    return chunks;
  }
}

/**
 * The "layered, title-separate" composite strategy: produces a "title" layer
 * chunk, a "note" layer chunk (whole body), and "paragraph" layer chunks, all
 * for a single note in one call. Implemented as a thin preset on top of
 * LayeredChunker.
 */
export interface TitleAwareLayeredChunkerOptions {
  /**
   * Passed through to the paragraph sub-strategy: when true, paragraph chunks
   * embed with the note title prepended (see ParagraphChunkerOptions). The
   * title and whole-note layers already contain the title and are unaffected.
   */
  contextualize?: boolean;
}

export class TitleAwareLayeredChunker extends LayeredChunker {
  constructor(options: TitleAwareLayeredChunkerOptions = {}) {
    super(
      [
        new TitleOnlyChunker(),
        new WholeNoteChunker(),
        new ParagraphChunker({ contextualize: options.contextualize }),
      ],
      "title-aware-layered",
    );
  }
}
