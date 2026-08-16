import { TransformersEmbeddingProvider, type FusionMethod, type ModelLoadProgress } from "sixdegrees";
import { demoNotes, type Note } from "./notes.js";
import {
  strategyOptions,
  modelOptions,
  similarityOptions,
  fusionOptions,
  DEFAULT_WINDOW_SIZE,
  DEFAULT_OVERLAP,
} from "./strategies.js";
import { LabChunkIndex, type LabSearchResult, type SimilarityMetric } from "./chunkIndex.js";

const notesById = new Map<string, Note>(demoNotes.map((n) => [n.id, n]));

// ---- state ----
let strategyId = strategyOptions[0]!.id;
let modelId = modelOptions[0]!.id;
let similarity: SimilarityMetric = "cosine";
let fusionMethod: FusionMethod = "max";
let layerWeights: Record<string, number> = {};
let windowSize = DEFAULT_WINDOW_SIZE;
let overlap = DEFAULT_OVERLAP;
let query = "";

let index = new LabChunkIndex();
let embedder = new TransformersEmbeddingProvider({ modelId });
let building = false;
let buildError: string | null = null;
let searching = false;
let results: LabSearchResult[] = [];
let indexReady = false;
let buildProgress: { done: number; total: number } | null = null;
let modelLoadProgress: ModelLoadProgress | null = null;

// ---- DOM scaffold ----
const app = document.getElementById("app")!;
app.innerHTML = `
  <h1>sixdegrees lab</h1>
  <p class="subtitle">Experiment with chunking strategy &times; similarity metric &times; score fusion over ${demoNotes.length} demo notes.</p>

  <div class="layout">
    <div class="main-col">
      <div class="search-row">
        <input type="text" id="query-input" placeholder="Search notes..." autocomplete="off" />
        <button id="search-btn">Search</button>
      </div>

      <div class="controls">
        <div class="control">
          <label for="strategy-select">Chunking strategy</label>
          <select id="strategy-select"></select>
        </div>
        <div class="control">
          <label for="model-select">Embedding model</label>
          <select id="model-select"></select>
        </div>
        <div class="control">
          <label for="similarity-select">Similarity metric</label>
          <select id="similarity-select"></select>
        </div>
        <div class="control">
          <label for="fusion-select">Fusion method</label>
          <select id="fusion-select"></select>
        </div>
        <div class="weights-panel" id="weights-panel" hidden></div>
        <div class="window-panel" id="window-panel" hidden></div>
      </div>

      <div class="status-row" id="status-row"></div>

      <div class="results" id="results"></div>
    </div>

    <aside class="notes-sidebar">
      <h2>Notes <span class="notes-count">(${demoNotes.length})</span></h2>
      <input type="text" id="notes-filter" placeholder="Filter notes..." autocomplete="off" />
      <div class="notes-list" id="notes-list"></div>
    </aside>
  </div>
`;

const queryInput = document.getElementById("query-input") as HTMLInputElement;
const searchBtn = document.getElementById("search-btn") as HTMLButtonElement;
const strategySelect = document.getElementById("strategy-select") as HTMLSelectElement;
const modelSelect = document.getElementById("model-select") as HTMLSelectElement;
const similaritySelect = document.getElementById("similarity-select") as HTMLSelectElement;
const fusionSelect = document.getElementById("fusion-select") as HTMLSelectElement;
const weightsPanel = document.getElementById("weights-panel") as HTMLDivElement;
const windowPanel = document.getElementById("window-panel") as HTMLDivElement;
const statusRow = document.getElementById("status-row") as HTMLDivElement;
const resultsEl = document.getElementById("results") as HTMLDivElement;
const notesFilterInput = document.getElementById("notes-filter") as HTMLInputElement;
const notesListEl = document.getElementById("notes-list") as HTMLDivElement;

let notesFilter = "";
const expandedNoteIds = new Set<string>();

function populateSelect(select: HTMLSelectElement, opts: { id: string; label: string }[]) {
  select.innerHTML = opts.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
}

populateSelect(strategySelect, strategyOptions);
populateSelect(modelSelect, modelOptions);
populateSelect(similaritySelect, similarityOptions as unknown as { id: string; label: string }[]);
populateSelect(fusionSelect, fusionOptions as unknown as { id: string; label: string }[]);
strategySelect.value = strategyId;
modelSelect.value = modelId;
similaritySelect.value = similarity;
fusionSelect.value = fusionMethod;

function currentStrategy() {
  return strategyOptions.find((s) => s.id === strategyId)!;
}

// ---- rendering ----
function renderStatus() {
  if (buildError) {
    statusRow.innerHTML = `<span class="error">Index build failed: ${escapeHtml(buildError)}</span>`;
    return;
  }
  if (building) {
    let progressText: string;
    // buildProgress starts at {done: 0, total} before the first embed() call
    // resolves, so a real model download (reported via modelLoadProgress)
    // would otherwise be hidden behind a stuck "0/N" readout. Prefer showing
    // model-load progress until at least one chunk has actually been embedded.
    if (buildProgress && buildProgress.done > 0) {
      progressText = `embedding chunks ${buildProgress.done}/${buildProgress.total}`;
    } else if (modelLoadProgress) {
      progressText = describeModelLoadProgress(modelLoadProgress);
    } else if (buildProgress) {
      progressText = `embedding chunks ${buildProgress.done}/${buildProgress.total}`;
    } else {
      progressText = "loading model...";
    }
    statusRow.innerHTML = `<span class="spinner"></span><span>Building index — ${escapeHtml(progressText)}...</span>`;
    return;
  }
  if (searching) {
    statusRow.innerHTML = `<span class="spinner"></span><span>Searching...</span>`;
    return;
  }
  if (indexReady) {
    statusRow.textContent = `Index ready (${currentStrategy().label}, ${modelId}).`;
  } else {
    statusRow.textContent = "";
  }
}

function describeModelLoadProgress(p: ModelLoadProgress): string {
  if (p.status === "progress" && typeof p.progress === "number") {
    const pct = p.progress.toFixed(0);
    const file = p.file ? ` ${p.file}` : "";
    return `downloading model${file} ${pct}%`;
  }
  if (p.status === "initiate") {
    return `fetching model file${p.file ? ` ${p.file}` : ""}`;
  }
  if (p.status === "done") {
    return `model file ready${p.file ? ` ${p.file}` : ""}`;
  }
  if (p.status === "ready") {
    return "model ready, embedding...";
  }
  return p.status;
}

function renderWindowPanel() {
  const strat = currentStrategy();
  windowPanel.hidden = !strat.hasWindowParams;
  if (!strat.hasWindowParams) return;

  windowPanel.innerHTML = `
    <div class="weight-slider">
      <label for="window-size-slider">Window size (sentences) <output id="window-size-out">${windowSize}</output></label>
      <input type="range" id="window-size-slider" min="1" max="8" step="1" value="${windowSize}" />
    </div>
    <div class="weight-slider">
      <label for="overlap-slider">Overlap (sentences) <output id="overlap-out">${overlap}</output></label>
      <input type="range" id="overlap-slider" min="0" max="${windowSize - 1}" step="1" value="${overlap}" />
    </div>
  `;

  const windowSlider = document.getElementById("window-size-slider") as HTMLInputElement;
  const overlapSlider = document.getElementById("overlap-slider") as HTMLInputElement;
  const windowOut = document.getElementById("window-size-out") as HTMLOutputElement;
  const overlapOut = document.getElementById("overlap-out") as HTMLOutputElement;

  windowSlider.addEventListener("input", () => {
    windowOut.textContent = windowSlider.value;
  });
  overlapSlider.addEventListener("input", () => {
    overlapOut.textContent = overlapSlider.value;
  });

  windowSlider.addEventListener("change", () => {
    windowSize = parseInt(windowSlider.value, 10);
    if (overlap >= windowSize) overlap = windowSize - 1;
    renderWindowPanel();
    void rebuildIndex().then(() => runSearch());
  });
  overlapSlider.addEventListener("change", () => {
    overlap = parseInt(overlapSlider.value, 10);
    void rebuildIndex().then(() => runSearch());
  });
}

function renderWeightsPanel() {
  const layers = currentStrategy().layers;
  const showWeights = fusionMethod === "weighted-sum" && layers.length > 1;
  weightsPanel.hidden = !showWeights;
  if (!showWeights) return;

  weightsPanel.innerHTML = layers
    .map((layer) => {
      const w = layerWeights[layer] ?? 1;
      return `
        <div class="weight-slider">
          <label for="weight-${layer}">${layer} weight <output id="weight-${layer}-out">${w.toFixed(1)}</output></label>
          <input type="range" id="weight-${layer}" min="0" max="3" step="0.1" value="${w}" data-layer="${layer}" />
        </div>
      `;
    })
    .join("");

  weightsPanel.querySelectorAll<HTMLInputElement>("input[type=range]").forEach((slider) => {
    slider.addEventListener("input", () => {
      const layer = slider.dataset.layer!;
      const val = parseFloat(slider.value);
      layerWeights[layer] = val;
      const out = document.getElementById(`weight-${layer}-out`)!;
      out.textContent = val.toFixed(1);
      runSearch();
    });
  });
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

function snippet(text: string, max = 160): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > max ? clean.slice(0, max).trimEnd() + "…" : clean;
}

function renderResults() {
  if (!indexReady) {
    resultsEl.innerHTML = "";
    return;
  }
  if (results.length === 0) {
    resultsEl.innerHTML = `<div class="empty-state">No results${query ? " for this query" : " yet — type a query and hit Search"}.</div>`;
    return;
  }

  resultsEl.innerHTML = results
    .map((r) => {
      const note = notesById.get(r.noteId);
      if (!note) return "";

      // best score per layer, for badges
      const bestPerLayer = new Map<string, number>();
      for (const sc of r.matchedChunks) {
        const prev = bestPerLayer.get(sc.chunk.layer);
        if (prev === undefined || sc.score > prev) bestPerLayer.set(sc.chunk.layer, sc.score);
      }

      const badges = Array.from(bestPerLayer.entries())
        .map(([layer, score]) => `<span class="badge">${escapeHtml(layer)}: ${score.toFixed(2)}</span>`)
        .join("");

      return `
        <div class="result-card">
          <div class="result-header">
            <h3>${escapeHtml(note.title)}</h3>
            <span class="result-score">score ${r.score.toFixed(3)}</span>
          </div>
          <p class="result-snippet">${escapeHtml(snippet(note.body))}</p>
          <div class="badges">${badges}</div>
        </div>
      `;
    })
    .join("");
}

// ---- notes sidebar ----
function renderNotesList() {
  const filter = notesFilter.trim().toLowerCase();
  const filtered = demoNotes.filter(
    (n) => !filter || n.title.toLowerCase().includes(filter) || n.body.toLowerCase().includes(filter),
  );

  if (filtered.length === 0) {
    notesListEl.innerHTML = `<div class="empty-state">No notes match this filter.</div>`;
    return;
  }

  notesListEl.innerHTML = filtered
    .map((n) => {
      const expanded = expandedNoteIds.has(n.id);
      return `
        <div class="note-item ${expanded ? "expanded" : ""}" data-note-id="${n.id}">
          <div class="note-item-header">
            <h4>${escapeHtml(n.title)}</h4>
            <span class="note-date">${escapeHtml(n.createdAt)}</span>
          </div>
          <p class="note-item-body">${expanded ? escapeHtml(n.body).replace(/\n/g, "<br>") : escapeHtml(snippet(n.body, 90))}</p>
        </div>
      `;
    })
    .join("");

  notesListEl.querySelectorAll<HTMLDivElement>(".note-item").forEach((el) => {
    el.addEventListener("click", () => {
      const id = el.dataset.noteId!;
      if (expandedNoteIds.has(id)) {
        expandedNoteIds.delete(id);
      } else {
        expandedNoteIds.add(id);
      }
      renderNotesList();
    });
  });
}

notesFilterInput.addEventListener("input", () => {
  notesFilter = notesFilterInput.value;
  renderNotesList();
});

renderNotesList();

// ---- index building ----
async function rebuildIndex() {
  building = true;
  buildError = null;
  indexReady = false;
  buildProgress = null;
  results = [];
  renderStatus();
  renderResults();

  try {
    modelLoadProgress = null;
    embedder = new TransformersEmbeddingProvider({
      modelId,
      onModelLoadProgress: (p) => {
        modelLoadProgress = p;
        renderStatus();
      },
    });
    index = new LabChunkIndex();
    const chunker = currentStrategy().create({ windowSize, overlap });
    await index.build(demoNotes, chunker, embedder, (done, total) => {
      buildProgress = { done, total };
      renderStatus();
    });
    indexReady = true;
  } catch (err) {
    buildError = err instanceof Error ? err.message : String(err);
  } finally {
    building = false;
    buildProgress = null;
    modelLoadProgress = null;
    renderStatus();
    renderResults();
  }
}

// ---- search ----
async function runSearch() {
  if (!indexReady || building) return;
  if (!query.trim()) {
    results = [];
    renderResults();
    return;
  }
  searching = true;
  renderStatus();
  try {
    results = await index.search(query, {
      similarity,
      fusion: {
        method: fusionMethod,
        layerWeights: fusionMethod === "weighted-sum" ? layerWeights : undefined,
      },
      topK: 10,
    });
  } finally {
    searching = false;
    renderStatus();
    renderResults();
  }
}

// ---- event wiring ----
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    query = queryInput.value;
    void runSearch();
  }
});
searchBtn.addEventListener("click", () => {
  query = queryInput.value;
  void runSearch();
});

strategySelect.addEventListener("change", () => {
  strategyId = strategySelect.value;
  layerWeights = {};
  windowSize = DEFAULT_WINDOW_SIZE;
  overlap = DEFAULT_OVERLAP;
  renderWeightsPanel();
  renderWindowPanel();
  void rebuildIndex().then(() => runSearch());
});

modelSelect.addEventListener("change", () => {
  modelId = modelSelect.value;
  void rebuildIndex().then(() => runSearch());
});

similaritySelect.addEventListener("change", () => {
  similarity = similaritySelect.value as SimilarityMetric;
  void runSearch();
});

fusionSelect.addEventListener("change", () => {
  fusionMethod = fusionSelect.value as FusionMethod;
  renderWeightsPanel();
  void runSearch();
});

// ---- init ----
renderWeightsPanel();
renderWindowPanel();
void rebuildIndex();
