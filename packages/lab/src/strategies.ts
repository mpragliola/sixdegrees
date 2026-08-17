import {
  WholeNoteChunker,
  ParagraphChunker,
  SentenceWindowChunker,
  TitleAwareLayeredChunker,
  type ChunkingStrategy,
} from "sixdegrees";

export interface StrategyParams {
  windowSize: number;
  overlap: number;
  /** Prepend the note title to paragraph/sentence-window embedding text. */
  contextualize?: boolean;
}

export interface StrategyOption {
  id: string;
  label: string;
  /** Layers this strategy can produce chunks in, for driving the weight-slider UI. */
  layers: string[];
  /** True if this strategy's chunker takes { windowSize, overlap }, for driving the window/overlap slider UI. */
  hasWindowParams?: boolean;
  /** True if this strategy supports the contextualize (title-prefixed embedding) toggle. */
  hasContextualize?: boolean;
  create: (params?: StrategyParams) => ChunkingStrategy;
}

export const DEFAULT_WINDOW_SIZE = 3;
export const DEFAULT_OVERLAP = 1;

export const strategyOptions: StrategyOption[] = [
  {
    id: "whole-note",
    label: "Whole note",
    layers: ["note"],
    create: () => new WholeNoteChunker(),
  },
  {
    id: "paragraph",
    label: "Paragraphs",
    layers: ["paragraph"],
    hasContextualize: true,
    create: (params) => new ParagraphChunker({ contextualize: params?.contextualize ?? false }),
  },
  {
    id: "sentence-window",
    label: "Sentence windows",
    layers: ["sentence-window"],
    hasWindowParams: true,
    hasContextualize: true,
    create: (params) =>
      new SentenceWindowChunker({
        windowSize: params?.windowSize ?? DEFAULT_WINDOW_SIZE,
        overlap: params?.overlap ?? DEFAULT_OVERLAP,
        contextualize: params?.contextualize ?? false,
      }),
  },
  {
    id: "layered",
    label: "Layered (title + note + paragraphs)",
    layers: ["title", "note", "paragraph"],
    hasContextualize: true,
    create: (params) => new TitleAwareLayeredChunker({ contextualize: params?.contextualize ?? false }),
  },
];

export const modelOptions = [
  { id: "Xenova/all-MiniLM-L6-v2", label: "all-MiniLM-L6-v2 (fast, 384-dim)" },
  { id: "Xenova/all-mpnet-base-v2", label: "all-mpnet-base-v2 (slower, 768-dim)" },
];

export const similarityOptions = [
  { id: "cosine", label: "Cosine" },
  { id: "bm25", label: "BM25 (lexical)" },
] as const;

export const fusionOptions = [
  { id: "max", label: "Max" },
  { id: "weighted-sum", label: "Weighted sum" },
] as const;
