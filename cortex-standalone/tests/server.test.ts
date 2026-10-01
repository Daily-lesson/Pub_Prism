import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as os from 'node:os';
import * as path from 'node:path';
import express from 'express';
import { loadRegistryFile } from '../core/index';
import { createAdvisorRouter, type AdvisorRouterOptions } from '../server/router';
import { createStandaloneApp } from '../server/standalone';

const REGISTRY_PATH = path.join(__dirname, '..', 'registry', 'examples', 'ops-dashboard.registry.json');

interface Running { url: string; close(): Promise<void> }

async function listen(app: express.Express): Promise<Running> {
  const server = http.createServer(app);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as { port: number };
  return { url: `http://127.0.0.1:${port}`, close: () => new Promise((r) => server.close(() => r())) };
}

function appWith(overrides: Partial<AdvisorRouterOptions> = {}): express.Express {
  const app = express();
  app.use('/api/advisor', createAdvisorRouter({ registry: loadRegistryFile(REGISTRY_PATH), ...overrides }));
  return app;
}

async function post(url: string, body: unknown, raw = false): Promise<{ status: number; json: any; text: string }> {
  const res = await fetch(`${url}/api/advisor`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: raw ? (body as string) : JSON.stringify(body),
  });
  const text = await res.text();
  let json: any = null;
  try { json = JSON.parse(text); } catch { /* not json */ }
  return { status: res.status, json, text };
}

// The standalone app auto-detects the committed example model (examples/demo-dashboard/cortex/
// model/ledger.json). A from-source checkout without the bundle runs keyword-only; the shipped
// package runs the model. Both are legitimate, so these cases assert whichever the checkout has
// rather than assuming one — a test that presumed "no artifacts" broke the moment the bundle landed.
const EXAMPLE_LEDGER = path.join(__dirname, '..', 'examples', 'demo-dashboard', 'cortex', 'model', 'ledger.json');
const HAVE_MODEL = fs.existsSync(EXAMPLE_LEDGER);
const EXAMPLE_MODEL_VERSION: string | null = HAVE_MODEL ? (JSON.parse(fs.readFileSync(EXAMPLE_LEDGER, 'utf8')) as { version: string }).version : null;

describe(`server — standalone app, example registry (${HAVE_MODEL ? 'example model present' : 'no artifacts'})`, () => {
  let s: Running;
  beforeAll(async () => { s = await listen(createStandaloneApp()); });
  afterAll(async () => { await s.close(); });

  it(`answers a howto query on the ${HAVE_MODEL ? 'cortex' : 'keyword'} rung with the contract shape and chips`, async () => {
    const r = await post(s.url, { query: 'how do i raise a ticket' });
    expect(r.status).toBe(200);
    expect(Object.keys(r.json).sort()).toEqual(['answer', 'chips', 'evidence', 'intent', 'intentConf', 'ladderRung', 'modelVersion'].sort());
    expect(r.json.intent).toBe('create_ticket');
    expect(r.json.ladderRung).toBe(HAVE_MODEL ? 'cortex' : 'keyword');
    expect(r.json.modelVersion).toBe(EXAMPLE_MODEL_VERSION);
    expect(typeof r.json.answer).toBe('string');
    expect(r.json.answer.length).toBeGreaterThan(0);
    expect(Array.isArray(r.json.chips)).toBe(true);
    expect(r.json.chips.length).toBeGreaterThan(0);
    expect(r.json.chips[0]).toEqual(expect.objectContaining({ label: expect.any(String), target: expect.anything() }));
  });

  it('rejects an empty, oversize, or missing query with 400 INVALID_QUERY', async () => {
    for (const body of [{ query: '' }, { query: '   ' }, { query: 'x'.repeat(2001) }, {}, { query: 42 }, { query: null }, [], 'just a string']) {
      const r = await post(s.url, body);
      expect(r.status, JSON.stringify(body).slice(0, 40)).toBe(400);
      expect(r.json).toEqual({ error: 'INVALID_QUERY' });
    }
    // malformed JSON is a bad query too, never a stack trace
    const bad = await post(s.url, '{"query": ', true);
    expect(bad.status).toBe(400);
    expect(bad.json).toEqual({ error: 'INVALID_QUERY' });
    // a 2000-char query is the ceiling, not over it
    const edge = await post(s.url, { query: 'x'.repeat(2000) });
    expect(edge.status).toBe(200);
  });

  it('ignores unknown body fields and never echoes them', async () => {
    const r = await post(s.url, { query: 'hi', evil: 'ignored-marker-7f3a' });
    expect(r.status).toBe(200);
    expect(r.text).not.toContain('ignored-marker-7f3a');
    expect(r.text).not.toContain('evil');
  });

  it('a status intent with no provider returns the registry unavailable copy — never a number', async () => {
    const r = await post(s.url, { query: 'how many tickets are open' });
    expect(r.status).toBe(200);
    expect(r.json.intent).toBe('open_tickets_status');
    expect(r.json.answer).toContain("I can't read the live ticket queue from here");
  });

  it('model routes serve exactly the ledger-referenced files when the example model is present, else 404', async () => {
    const ledgerRes = await fetch(`${s.url}/api/advisor/model/ledger.json`);
    if (!HAVE_MODEL) {
      expect(ledgerRes.status).toBe(404);
      expect(await ledgerRes.json()).toEqual({ error: 'NOT_FOUND' });
      return;
    }
    expect(ledgerRes.status).toBe(200);
    const ledger = (await ledgerRes.json()) as { version: string; onnx: { file: string } };
    expect(ledger.version).toBe(EXAMPLE_MODEL_VERSION);
    const onnx = await fetch(`${s.url}/api/advisor/model/${ledger.onnx.file}`);
    expect(onnx.status).toBe(200);
    // never a file the ledger does not name, even one that exists on disk
    for (const p of ['/api/advisor/model/a.onnx', '/api/advisor/model/../package.json', '/api/advisor/model/ops-dashboard-0.1.0.onnx.bak']) {
      const res = await fetch(`${s.url}${p}`);
      expect(res.status, p).toBe(404);
    }
  });

  it('model and runtime routes 404 on a router built with neither directory', async () => {
    // Built explicitly without artifactsDir/runtimeDir: the standalone app picks up `runtime/`
    // automatically when the package ships one, so this must not depend on the checkout.
    const bare = await listen(appWith({}));
    try {
      for (const p of ['/api/advisor/model/ledger.json', '/api/advisor/runtime/runtime-manifest.json', '/api/advisor/runtime/ort.wasm.min.mjs']) {
        const res = await fetch(`${bare.url}${p}`);
        expect(res.status, p).toBe(404);
        expect(await res.json()).toEqual({ error: 'NOT_FOUND' });
      }
    } finally { await bare.close(); }
  });

  it('serves the example dashboard statically', async () => {
    const res = await fetch(`${s.url}/registry/examples/ops-dashboard.registry.json`);
    expect(res.status).toBe(200);
  });
});

describe('server — asset allowlist', () => {
  let dir: string;
  let s: Running;
  beforeAll(async () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cortex-artifacts-'));
    fs.writeFileSync(path.join(dir, 'ledger.json'), JSON.stringify({
      version: '9.9.9', app: 'test',
      onnx: { file: 'a.onnx', sha256: 'x', bytes: 4 },
      tokenizer: { file: 'a.tokenizer.json', sha256: 'x' },
      labels: { file: '../package.json', sha256: 'x' }, // a path-shaped ref is never allowlisted
    }));
    fs.writeFileSync(path.join(dir, 'a.onnx'), Buffer.from([0, 1, 2, 3]));
    fs.writeFileSync(path.join(dir, 'secret.txt'), 'nope');
    const rt = path.join(dir, 'rt');
    fs.mkdirSync(rt);
    fs.writeFileSync(path.join(rt, 'runtime-manifest.json'), JSON.stringify({
      entry: { file: 'ort.wasm.min.mjs' }, loader: { file: 'glue.mjs' }, wasm: { file: 'glue.wasm' },
    }));
    fs.writeFileSync(path.join(rt, 'ort.wasm.min.mjs'), 'export const x = 1;');
    fs.writeFileSync(path.join(rt, 'glue.wasm'), Buffer.from([0x00, 0x61, 0x73, 0x6d]));
    fs.writeFileSync(path.join(rt, 'private.mjs'), 'export const y = 2;');
    s = await listen(appWith({ artifactsDir: dir, runtimeDir: rt }));
  });
  afterAll(async () => { await s.close(); fs.rmSync(dir, { recursive: true, force: true }); });

  it('serves an allowlisted artifact with the right content type and immutable caching', async () => {
    const res = await fetch(`${s.url}/api/advisor/model/a.onnx`);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('application/octet-stream');
    expect(res.headers.get('cache-control')).toBe('public, max-age=31536000, immutable');
    expect(Buffer.from(await res.arrayBuffer())).toEqual(Buffer.from([0, 1, 2, 3]));
    const ledger = await fetch(`${s.url}/api/advisor/model/ledger.json`);
    expect(ledger.status).toBe(200);
    expect(ledger.headers.get('content-type')).toMatch(/^application\/json/);
    expect(((await ledger.json()) as { version: string }).version).toBe('9.9.9');
  });

  it('404s a file that exists on disk but is not in the allowlist, and every path-shaped name', async () => {
    for (const p of ['secret.txt', '..%2Fpackage.json', '..%2F..%2Fpackage.json', '%2e%2e%2fpackage.json', 'package.json', 'a.tokenizer.json', 'nope.onnx']) {
      const res = await fetch(`${s.url}/api/advisor/model/${p}`);
      expect(res.status, p).toBe(404);
      expect(await res.json()).toEqual({ error: 'NOT_FOUND' });
    }
    // `/model/.` is collapsed by URL normalisation before routing — Express's own 404, still a 404
    expect((await fetch(`${s.url}/api/advisor/model/.`)).status).toBe(404);
    // the path-shaped labels ref was dropped from the allowlist at read time
    expect((await fetch(`${s.url}/api/advisor/model/package.json`)).status).toBe(404);
  });

  it('runtime route: allowlisted .mjs/.wasm served with their types; a stray .mjs is 404', async () => {
    const mjs = await fetch(`${s.url}/api/advisor/runtime/ort.wasm.min.mjs`);
    expect(mjs.status).toBe(200);
    expect(mjs.headers.get('content-type')).toMatch(/^text\/javascript/);
    expect(mjs.headers.get('cache-control')).toBe('public, max-age=31536000, immutable');
    const wasm = await fetch(`${s.url}/api/advisor/runtime/glue.wasm`);
    expect(wasm.status).toBe(200);
    expect(wasm.headers.get('content-type')).toBe('application/wasm');
    // named in the manifest but missing on disk
    expect((await fetch(`${s.url}/api/advisor/runtime/glue.mjs`)).status).toBe(404);
    expect((await fetch(`${s.url}/api/advisor/runtime/private.mjs`)).status).toBe(404);
  });
});

describe('server — host providers', () => {
  it('context(req) is passed through to status; a string is used verbatim; undefined yields the unavailable copy', async () => {
    const status = vi.fn((ctx: unknown, intentId: string) => {
      if (intentId === 'open_tickets_status' && (ctx as { user: string }).user === 'ada') return 'There are 7 tickets open for ada.';
      return undefined;
    });
    const context = vi.fn((req: express.Request) => ({ user: String(req.headers['x-user'] ?? 'anon') }));
    const s = await listen(appWith({ status, context }));
    try {
      const res = await fetch(`${s.url}/api/advisor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-user': 'ada' },
        body: JSON.stringify({ query: 'how many tickets are open' }),
      });
      const json = await res.json() as any;
      expect(res.status).toBe(200);
      expect(json.intent).toBe('open_tickets_status');
      expect(json.answer).toBe('There are 7 tickets open for ada.');
      expect(context).toHaveBeenCalledTimes(1);
      expect(status).toHaveBeenCalledWith({ user: 'ada' }, 'open_tickets_status', expect.any(Array));

      const anon = await post(s.url, { query: 'how many tickets are open' });
      expect(anon.json.answer).toContain("I can't read the live ticket queue from here");
      expect(anon.json.answer).not.toMatch(/\d/);
      expect(status).toHaveBeenLastCalledWith({ user: 'anon' }, 'open_tickets_status', expect.any(Array));

      // a howto query never consults the status provider
      status.mockClear();
      await post(s.url, { query: 'how do i raise a ticket' });
      expect(status).not.toHaveBeenCalled();
    } finally { await s.close(); }
  });

  it('async status providers are awaited', async () => {
    const s = await listen(appWith({ status: async () => 'async answer' }));
    try {
      const r = await post(s.url, { query: 'is everything running' });
      expect(r.json.intent).toBe('system_health');
      expect(r.json.answer).toBe('async answer');
    } finally { await s.close(); }
  });

  it('isAllowed(ctx, need) false drops the gated chip and receives the request context', async () => {
    const isAllowed = vi.fn((ctx: unknown, need: string) => !((ctx as { role: string }).role === 'viewer' && need === 'tickets'));
    const s = await listen(appWith({ isAllowed, context: (req) => ({ role: String(req.headers['x-role'] ?? 'admin') }) }));
    try {
      const viewer = await fetch(`${s.url}/api/advisor`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'x-role': 'viewer' },
        body: JSON.stringify({ query: 'how do i raise a ticket' }),
      });
      const vj = await viewer.json() as any;
      expect(vj.intent).toBe('create_ticket');
      expect(vj.chips).toEqual([]); // both create_ticket links need "tickets"
      expect(isAllowed).toHaveBeenCalledWith({ role: 'viewer' }, 'tickets');

      const admin = await post(s.url, { query: 'how do i raise a ticket' });
      expect(admin.json.chips.map((c: { label: string }) => c.label)).toEqual(['Tickets', 'New ticket']);
    } finally { await s.close(); }
  });

  it('concurrent requests each see their own context', async () => {
    const status = async (ctx: unknown) => {
      await new Promise((r) => setTimeout(r, 20));
      return `user=${(ctx as { u: string }).u}`;
    };
    const s = await listen(appWith({ status, context: (req) => ({ u: String(req.headers['x-u']) }) }));
    try {
      const results = await Promise.all(['a', 'b', 'c', 'd'].map(async (u) => {
        const res = await fetch(`${s.url}/api/advisor`, {
          method: 'POST', headers: { 'Content-Type': 'application/json', 'x-u': u },
          body: JSON.stringify({ query: 'is everything running' }),
        });
        return ((await res.json()) as any).answer as string;
      }));
      expect(results).toEqual(['user=a', 'user=b', 'user=c', 'user=d']);
    } finally { await s.close(); }
  });
});

describe('server — relative artifactsDir/runtimeDir (review MF-2)', () => {
  it('the README example shape (cwd-relative dirs) serves the asset routes instead of 500ing', async () => {
    const rel = path.relative(process.cwd(), path.join(__dirname, '..', 'examples', 'demo-dashboard', 'cortex', 'model'));
    const relRt = path.relative(process.cwd(), path.join(__dirname, '..', 'runtime'));
    const haveModel = fs.existsSync(path.join(rel, 'ledger.json'));
    const haveRt = fs.existsSync(path.join(relRt, 'runtime-manifest.json'));
    const s2 = await listen(appWith({ artifactsDir: haveModel ? rel : undefined, runtimeDir: haveRt ? relRt : undefined }));
    try {
      if (haveModel) {
        const r = await fetch(`${s2.url}/api/advisor/model/ledger.json`);
        expect(r.status).toBe(200);
        expect((await r.text()).startsWith('{')).toBe(true);
      }
      if (haveRt) {
        const r = await fetch(`${s2.url}/api/advisor/runtime/runtime-manifest.json`);
        expect(r.status).toBe(200);
      }
      // never an HTML error page with a filesystem path, whatever the checkout has
      const miss = await fetch(`${s2.url}/api/advisor/model/nope.onnx`);
      expect(miss.status).toBe(404);
      expect(await miss.text()).not.toContain('sendFile');
      expect(haveModel || haveRt, 'non-vacuity: at least one relative dir must exist in this checkout').toBe(true);
    } finally { await s2.close(); }
  });
});
