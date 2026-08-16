import type { Chunk, ChunkingStrategy } from "../types.js";

/**
 * One chunk containing just the title. Layer "title".
 * charStart/charEnd are offsets into `note.title`.
 */
export class TitleOnlyChunker implements ChunkingStrategy {
  readonly name = "title-only";

  chunk(note: { id: string; title: string; body: string }): Chunk[] {
    const text = note.title.trim();
    if (text.length === 0) return [];
    return [
      {
        id: `${note.id}::title::0`,
        noteId: note.id,
        layer: "title",
        text,
        charStart: 0,
        charEnd: text.length,
      },
    ];
  }
}
