import type { Chunk, ChunkingStrategy } from "../types.js";

export interface ParagraphChunkerOptions {
  /**
   * When true, each chunk gets `embeddingText = "${title}\n\n${text}"` so the
   * embedding carries note-level context. Skipped when the title is empty.
   * `text`/charStart/charEnd are unaffected. Default false.
   */
  contextualize?: boolean;
}

/**
 * Splits the note body on blank lines into paragraph chunks. Layer "paragraph".
 * charStart/charEnd are offsets into `note.body` (not title-prefixed).
 */
export class ParagraphChunker implements ChunkingStrategy {
  readonly name = "paragraph";
  private readonly contextualize: boolean;

  constructor(options: ParagraphChunkerOptions = {}) {
    this.contextualize = options.contextualize ?? false;
  }

  chunk(note: { id: string; title: string; body: string }): Chunk[] {
    const title = note.title.trim();
    const chunks: Chunk[] = [];
    const body = note.body;
    // Split on one-or-more blank lines (allowing trailing whitespace on the blank line).
    const paragraphRe = /\n\s*\n+/g;

    let lastIndex = 0;
    let paraIndex = 0;
    let match: RegExpExecArray | null;

    const pushParagraph = (raw: string, start: number, _end: number) => {
      const trimmed = raw.trim();
      if (trimmed.length === 0) return;
      const trimStart = start + raw.indexOf(trimmed);
      const chunk: Chunk = {
        id: `${note.id}::paragraph::${paraIndex}`,
        noteId: note.id,
        layer: "paragraph",
        text: trimmed,
        charStart: trimStart,
        charEnd: trimStart + trimmed.length,
      };
      if (this.contextualize && title.length > 0) {
        chunk.embeddingText = `${title}\n\n${trimmed}`;
      }
      chunks.push(chunk);
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
