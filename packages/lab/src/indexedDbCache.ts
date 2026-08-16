import type { EmbeddingCache } from "sixdegrees";

const DB_NAME = "sixdegrees-lab-embedding-cache";
const DB_VERSION = 1;
const STORE_NAME = "embeddings";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE_NAME);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
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
        resolve(value ? new Float32Array(value) : undefined);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async set(modelId: string, text: string, embedding: Float32Array): Promise<void> {
    const db = await this.db();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).put(Array.from(embedding), keyFor(modelId, text));
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  /** Clears all cached embeddings (e.g. for a "clear cache" dev action). */
  async clear(): Promise<void> {
    const db = await this.db();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
}
