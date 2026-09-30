/**
 * numpy-ts WebAssembly heap sizing.
 *
 * numpy-ts backs every NDArray with a fixed-size WebAssembly linear memory
 * pool (default 256 MiB).  Once this pool is full, allocations silently fall
 * back to JS-backed `TypedArray`s and *all* subsequent numpy kernels take the
 * slower copy-in / copy-out path for the rest of the process.  wanim creates
 * many short-lived NDArrays (points, rgba arrays, interpolation results) and
 * its hot loops rarely yield to the event loop, so the FinalizationRegistry
 * that normally reclaims WASM regions often cannot drain before the pool is
 * exhausted — the 256 MiB default is too small for even a handful of scenes.
 *
 * `configureWasm()` must run before the first array operation (numpy-ts throws
 * otherwise), so this module is imported as the very first dependency of
 * `src/index.ts` to initialise the heap before any other wanim module runs.
 *
 * The size can be overridden with the `WANIM_NUMPY_MAX_MEMORY` environment
 * variable (in MiB).  Browsers reserve the pool lazily, so a larger value does
 * not raise RSS up front; raise it further only for very large workloads.
 */
import { configureWasm } from "numpy-ts";

/** Default WASM heap size: 1 GiB (matches the web demo's explicit setting). */
const DEFAULT_MAX_MEMORY = 1024 * 1024 * 1024;

function envMaxMemory(): number | null {
  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
  const raw = env?.WANIM_NUMPY_MAX_MEMORY;
  if (raw === undefined || raw === "") return null;
  const megabytes = Number(raw);
  if (!Number.isFinite(megabytes) || megabytes <= 0) return null;
  return Math.floor(megabytes * 1024 * 1024);
}

let configured = false;

/**
 * Size the numpy-ts WASM heap once.  Safe to call repeatedly and safe to call
 * after numpy-ts has already been configured by an embedder (the latest call
 * before the first array operation wins; once the heap is live this is a
 * no-op because numpy-ts throws and the error is swallowed here).
 */
export function configure_numpy_wasm_heap(): void {
  if (configured) return;
  configured = true;
  try {
    configureWasm({ maxMemory: envMaxMemory() ?? DEFAULT_MAX_MEMORY });
  } catch {
    // The heap was already initialised by an earlier array operation; keep
    // whatever size is in effect instead of crashing the import.
  }
}

configure_numpy_wasm_heap();
