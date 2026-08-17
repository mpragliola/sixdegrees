/** Tiny prefixed/timestamped logger for tracing the lab's cache and embedding pipeline. */
const t0 = performance.now();

function elapsed(): string {
  return `+${(performance.now() - t0).toFixed(0)}ms`;
}

export function log(scope: string, msg: string, data?: unknown): void {
  if (data !== undefined) {
    console.log(`[lab:${scope}] ${elapsed()} ${msg}`, data);
  } else {
    console.log(`[lab:${scope}] ${elapsed()} ${msg}`);
  }
}
