import type { EmbeddingCache } from "sixdegrees";
import { log } from "./log.js";

const DB_NAME = "sixdegrees-lab-embedding-cache";
const DB_VERSION = 1;
const STORE_NAME = "embeddings";

function openDb(): Promise<IDBDatabase> {
  log("cache", `opening IndexedDB "${DB_NAME}" v${DB_VERSION}`);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      log("cache", "onupgradeneeded — creating object store (fresh/versioned DB)");
      req.result.createObjectStore(STORE_NAME);
    };
    req.onsuccess = () => {
      log("cache", "DB open succeeded");
      resolve(req.result);
    };
    req.onerror = () => {
      log("cache", "DB open FAILED", req.error);
      reject(req.error);
    };
  });
}

function keyFor(modelId: string, text: string): string {
  return `${modelId}::${text}`;
}

/**
 * IndexedDB-backed EmbeddingCache for the lab. Stores each embedding as a
 * plain number[] (structured-clone-friendly; Float32Array round-trips fine
 * too, but a plain array keeps the stored shape simple and portable).
 */
export class IndexedDbEmbeddingCache implements EmbeddingCache {
  private dbPromise: Promise<IDBDatabase> | null = null;
  private hits = 0;
  private misses = 0;

  private db(): Promise<IDBDatabase> {
    if (!this.dbPromise) this.dbPromise = openDb();
    return this.dbPromise;
  }

  async get(modelId: string, text: string): Promise<Float32Array | undefined> {
    const db = await this.db();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).get(keyFor(modelId, text));
      req.onsuccess = () => {
        const value = req.result as number[] | undefined;
        if (value) {
          this.hits++;
        } else {
          this.misses++;
        }
        if ((this.hits + this.misses) % 50 === 0) {
          log("cache", `progress: ${this.hits} hits / ${this.misses} misses so far`);
        }
        resolve(value ? new Float32Array(value) : undefined);
      };
      req.onerror = () => {
        log("cache", "get() FAILED", req.error);
        reject(req.error);
      };
    });
  }

  async set(modelId: string, text: string, embedding: Float32Array): Promise<void> {
    const db = await this.db();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).put(Array.from(embedding), keyFor(modelId, text));
      tx.oncomplete = () => resolve();
      tx.onerror = () => {
        log("cache", "set() FAILED", tx.error);
        reject(tx.error);
      };
    });
  }

  /** Clears all cached embeddings (e.g. for a "clear cache" dev action). */
  async clear(): Promise<void> {
    log("cache", "clear() — wiping all cached embeddings");
    this.hits = 0;
    this.misses = 0;
    const db = await this.db();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  /** Logs a final hit/miss summary; call once a build's embed calls are done. */
  logSummary(context: string): void {
    const total = this.hits + this.misses;
    const pct = total > 0 ? ((this.hits / total) * 100).toFixed(0) : "0";
    log("cache", `${context}: ${this.hits}/${total} hits (${pct}%), ${this.misses} misses`);
  }
}
