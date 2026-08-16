import type { Chunk, ChunkingStrategy } from "../types.js";

/**
 * One chunk per note: title + body concatenated. Layer "note".
 */
export class WholeNoteChunker implements ChunkingStrategy {
  readonly name = "whole-note";

  chunk(note: { id: string; title: string; body: string }): Chunk[] {
    const text = `${note.title}\n\n${note.body}`.trim();
    if (text.length === 0) return [];
    return [
      {
        id: `${note.id}::note::0`,
        noteId: note.id,
        layer: "note",
        text,
        charStart: 0,
        charEnd: text.length,
      },
    ];
  }
}
