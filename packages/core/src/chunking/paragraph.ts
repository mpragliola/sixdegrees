import type { Chunk, ChunkingStrategy } from "../types.js";

/**
 * Splits the note body on blank lines into paragraph chunks. Layer "paragraph".
 * charStart/charEnd are offsets into `note.body` (not title-prefixed).
 */
export class ParagraphChunker implements ChunkingStrategy {
  readonly name = "paragraph";

  chunk(note: { id: string; title: string; body: string }): Chunk[] {
    const chunks: Chunk[] = [];
    const body = note.body;
    // Split on one-or-more blank lines (allowing trailing whitespace on the blank line).
    const paragraphRe = /\n\s*\n+/g;

    let lastIndex = 0;
    let paraIndex = 0;
    let match: RegExpExecArray | null;

    const pushParagraph = (raw: string, start: number, end: number) => {
      const trimmed = raw.trim();
      if (trimmed.length === 0) return;
      const trimStart = start + raw.indexOf(trimmed);
      chunks.push({
        id: `${note.id}::paragraph::${paraIndex}`,
        noteId: note.id,
        layer: "paragraph",
        text: trimmed,
        charStart: trimStart,
        charEnd: trimStart + trimmed.length,
      });
      paraIndex++;
    };

    while ((match = paragraphRe.exec(body)) !== null) {
      const raw = body.slice(lastIndex, match.index);
      pushParagraph(raw, lastIndex, match.index);
      lastIndex = match.index + match[0].length;
    }
    pushParagraph(body.slice(lastIndex), lastIndex, body.length);

    return chunks;
  }
}
