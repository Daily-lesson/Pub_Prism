import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { decodeSlots, softmax, argmax, wordSplitWithOffsets, normalizeRegistry, loadArtifacts, classify, loadRegistryFile, createEngine, runRegressionGate, type Registry } from '../core/index';

const PKG = path.resolve(__dirname, '..');
const MODEL_DIR = path.join(PKG, 'examples', 'demo-dashboard', 'cortex', 'model');
const EXAMPLE = path.join(PKG, 'registry', 'examples', 'ops-dashboard.registry.json');
const HAVE_MODEL = fs.existsSync(path.join(MODEL_DIR, 'ledger.json'));

const R = normalizeRegistry({
  registryVersion: '1',
  app: { slug: 's', name: 'S' },
  slots: { region: { vocab: [{ id: 'north', label: 'North campus' }] }, period: { vocab: [{ id: 'week', label: 'this week' }] } },
  intents: [{ id: 'x', family: 'howto', label: 'X', templates: ['x'], answer: { steps: ['x'] } }],
} as Registry);

describe('decode (§4.3)', () => {
  it('softmax/argmax', () => {
    const p = softmax([1, 2, 3]);
    expect(p[0] + p[1] + p[2]).toBeCloseTo(1, 10);
    expect(argmax(p)).toBe(2);
    expect(argmax([0.1, 0.9, 0.5])).toBe(1);
  });

  it('BIO → spans, resolvedId by case-insensitive label, name change starts a new span', () => {
    const u = 'reports for North Campus this week ok';
    const toks = wordSplitWithOffsets(u);
    const labels = ['O', 'O', 'B-region', 'I-region', 'B-period', 'I-period', 'O'];
    expect(decodeSlots(u, labels, toks, R)).toEqual([
      { name: 'region', value: 'North Campus', start: 12, end: 24, resolvedId: 'north' },
      { name: 'period', value: 'this week', start: 25, end: 34, resolvedId: 'week' },
    ]);
    const changed = decodeSlots(u, ['O', 'O', 'I-region', 'I-period', 'O', 'O', 'O'], toks, R);
    expect(changed.map((s) => `${s.name}:${s.value}`)).toEqual(['region:North', 'period:Campus']);
    expect(changed[0].resolvedId).toBeUndefined();
  });
});

describe.skipIf(!HAVE_MODEL)('real model session (runs only when the example model exists)', () => {
  it('loads, verifies sha256, classifies a held-out utterance with the contract shape', async () => {
    const reg = loadRegistryFile(EXAMPLE);
    const art = await loadArtifacts(MODEL_DIR, reg);
    expect(art.ledger.version).toBeTruthy();
    const heldout = reg.heldout.find((h) => h.intent === 'find_reports') ?? reg.heldout[0];
    const r = await classify(heldout.utterance, art);
    expect(reg.intents.map((i) => i.id)).toContain(r.intent);
    expect(r.intentConf).toBeGreaterThan(0);
    expect(r.intentConf).toBeLessThanOrEqual(1);
    expect(r.modelVersion).toBe(art.ledger.version);
    expect(r.modelSha256).toBe(art.ledger.onnx.sha256);
    for (const s of r.slots) {
      expect(Object.keys(reg.slots)).toContain(s.name);
      expect(heldout.utterance.slice(s.start, s.end)).toBe(s.value);
    }
    const e = createEngine({ registry: reg, artifactsDir: MODEL_DIR });
    await e.warm();
    expect(e.mode).toBe('model');
    const a = await e.answer('hello there');
    expect(a.modelVersion).toBe(art.ledger.version);
  });

  it('the regression gate passes on the shipped example model', async () => {
    const reg = loadRegistryFile(EXAMPLE);
    const report = await runRegressionGate({ registry: reg, artifactsDir: MODEL_DIR });
    expect(report.reasons).toEqual([]);
    expect(report.passed).toBe(true);
    expect(report.artifact.sha256Verified).toBe(true);
    expect(report.dataset.deterministic).toBe(true);
  });
});
