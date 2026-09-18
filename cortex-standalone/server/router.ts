/**
 * The Express router (CONTRACTS §7).
 *
 * `POST /`            → `{ answer, chips, evidence, intent, intentConf, ladderRung, modelVersion }`
 * `GET /model/:file`  → an artifact named in `ledger.json` (allowlist), immutable-cached
 * `GET /runtime/:file`→ a file named in `runtime-manifest.json` (allowlist), immutable-cached
 *
 * Guarantees (§7.2): the router reads exactly one body field (`query`), never
 * logs the raw query, and never makes an outbound request. Rate limiting and
 * authentication are the HOST's job — mount this behind whatever the host
 * already uses for the rest of its API.
 *
 * One engine is created when the router is built and reused for every
 * request. The host's `status`/`isAllowed` providers receive the per-request
 * context (`opts.context(req)`) via AsyncLocalStorage, so a single shared
 * engine still sees who is asking on every call.
 */

import { AsyncLocalStorage } from 'node:async_hooks';
import * as fs from 'node:fs';
import * as path from 'node:path';
import express, { type NextFunction, type Request, type Response, type Router } from 'express';
import { createEngine, type Engine, type Registry } from '../core/index';

export type StatusSlot = { name: string; value: string; resolvedId?: string };

export interface AdvisorRouterOptions {
  registry: Registry;
  /** Directory holding `ledger.json` + the model artifacts. Absent ⇒ keyword-only, model routes 404. */
  artifactsDir?: string;
  /** Directory holding `runtime-manifest.json` + the browser runtime files. Absent ⇒ runtime routes 404. */
  runtimeDir?: string;
  /** Host status provider (§5.5). Returning `undefined` yields the registry's `unavailable` copy. */
  status?: (ctx: unknown, intentId: string, slots: readonly StatusSlot[]) => string | undefined | Promise<string | undefined>;
  /** Derives the per-request context handed to `status`/`isAllowed` (e.g. the signed-in user). */
  context?: (req: Request) => unknown;
  /** Link gate (§1.5). Absent ⇒ every link is allowed. */
  isAllowed?: (ctx: unknown, need: string) => boolean;
  confidenceGate?: number;
}

const QUERY_MAX = 2000;
const BODY_LIMIT = '16kb';
const IMMUTABLE = 'public, max-age=31536000, immutable';

export function contentTypeFor(file: string): string {
  if (file.endsWith('.onnx')) return 'application/octet-stream';
  if (file.endsWith('.wasm')) return 'application/wasm';
  if (file.endsWith('.mjs')) return 'text/javascript; charset=utf-8';
  if (file.endsWith('.json')) return 'application/json; charset=utf-8';
  return 'application/octet-stream';
}

/**
 * Allowlist = the manifest file itself + every `<ref>.file` it names. A
 * missing or malformed manifest yields an allowlist of just the manifest
 * name, and that name then 404s on the existence check — never a crash.
 */
export function readAllowlist(dir: string | undefined, manifestFile: string, refs: readonly string[]): Set<string> {
  const names = new Set<string>();
  if (!dir) return names;
  names.add(manifestFile);
  try {
    const manifest = JSON.parse(fs.readFileSync(path.join(dir, manifestFile), 'utf8')) as Record<string, unknown>;
    for (const ref of refs) {
      const entry = manifest[ref] as { file?: unknown } | undefined;
      if (entry && typeof entry.file === 'string' && entry.file === path.basename(entry.file)) names.add(entry.file);
    }
  } catch {
    // keep the manifest-only allowlist
  }
  return names;
}

function mountAssetRoute(router: Router, urlPath: string, dir: string | undefined, allowlist: Set<string>): void {
  router.get(urlPath, (req: Request, res: Response) => {
    const raw = String(req.params.file ?? '');
    // `..%2F…` decodes to a path shape; basename() strips it and the comparison rejects it —
    // the filesystem is never consulted for a name outside the allowlist.
    const file = path.basename(raw);
    if (!dir || file !== raw || !allowlist.has(file)) { res.status(404).json({ error: 'NOT_FOUND' }); return; }
    const p = path.join(dir, file);
    if (!fs.existsSync(p) || !fs.statSync(p).isFile()) { res.status(404).json({ error: 'NOT_FOUND' }); return; }
    res.setHeader('Content-Type', contentTypeFor(file));
    res.setHeader('Cache-Control', IMMUTABLE);
    res.sendFile(p);
  });
}

interface RequestScope { ctx: unknown }

export function createAdvisorRouter(opts: AdvisorRouterOptions): Router {
  // Directories are resolved ONCE, here: `res.sendFile` requires an absolute path, and the
  // README's own example passes cwd-relative ones — an unresolved dir 500'd every asset route
  // with Express's HTML error page (and the server's filesystem path in the stack).
  const artifactsDir = opts.artifactsDir ? path.resolve(opts.artifactsDir) : undefined;
  const runtimeDir = opts.runtimeDir ? path.resolve(opts.runtimeDir) : undefined;
  const scope = new AsyncLocalStorage<RequestScope>();
  const engine: Engine = createEngine({
    registry: opts.registry,
    artifactsDir,
    confidenceGate: opts.confidenceGate,
    status: opts.status
      ? (intentId, slots) => opts.status!(scope.getStore()?.ctx, intentId, slots)
      : undefined,
    isAllowed: opts.isAllowed
      ? (need) => opts.isAllowed!(scope.getStore()?.ctx, need)
      : undefined,
  });

  const router = express.Router();

  router.post('/', express.json({ limit: BODY_LIMIT, strict: true }), async (req: Request, res: Response) => {
    const body: unknown = req.body;
    const rawQuery = body && typeof body === 'object' && !Array.isArray(body) ? (body as { query?: unknown }).query : undefined;
    if (typeof rawQuery !== 'string') { res.status(400).json({ error: 'INVALID_QUERY' }); return; }
    const query = rawQuery.trim();
    if (query.length < 1 || query.length > QUERY_MAX) { res.status(400).json({ error: 'INVALID_QUERY' }); return; }

    const ctx = opts.context ? opts.context(req) : undefined;
    try {
      const r = await scope.run({ ctx }, () => engine.answer(query));
      res.json({
        answer: r.answerText,
        chips: r.chips,
        evidence: r.evidence,
        intent: r.intent,
        intentConf: r.intentConf,
        ladderRung: r.ladderRung,
        modelVersion: r.modelVersion,
      });
    } catch (err) {
      // The query itself is never logged (§7.2) — only the failure.
      console.error('[cortex] answer failed:', err instanceof Error ? err.message : String(err));
      res.status(500).json({ error: 'ANSWER_FAILED' });
    }
  });

  // Body-parser failures (malformed JSON, oversize body, wrong type) are a bad query, not a crash.
  router.use((err: unknown, _req: Request, res: Response, next: NextFunction) => {
    if (res.headersSent) { next(err); return; }
    const e = err as { type?: string; status?: number } | null;
    if (e && (typeof e.type === 'string' && e.type.startsWith('entity.') || e.status === 400 || e.status === 413)) {
      res.status(400).json({ error: 'INVALID_QUERY' });
      return;
    }
    next(err);
  });

  mountAssetRoute(router, '/model/:file', artifactsDir,
    readAllowlist(artifactsDir, 'ledger.json', ['onnx', 'tokenizer', 'labels']));
  mountAssetRoute(router, '/runtime/:file', runtimeDir,
    readAllowlist(runtimeDir, 'runtime-manifest.json', ['entry', 'loader', 'wasm']));

  return router;
}
