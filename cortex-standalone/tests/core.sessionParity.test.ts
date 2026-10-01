/**
 * Cross-runtime parity (docs/CONTRACTS.md §4): the Node session (`onnxruntime-node`) must
 * reproduce the Python reference (`onnxruntime` via train/gen_session_parity_fixture.py) on the
 * SAME committed artifact — identical intent, confidence within ±0.01, identical slot spans.
 *
 * The fixture is generated against a specific artifact bundle; if the example model is retrained
 * the fixture MUST be regenerated (the test pins this by comparing the fixture's artifactsDir
 * and by re-verifying the ledger's sha256 through loadArtifacts). Runs only when the example
 * model exists — a from-source checkout without the bundle skips honestly, it does not pass.
 */
import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { classify, loadArtifacts, loadRegistryFile } from '../core/index';

const PKG = path.resolve(__dirname, '..');
const FIXTURE = path.join(PKG, 'tests', 'fixtures', 'session.parity.json');
const EXAMPLE = path.join(PKG, 'registry', 'examples', 'ops-dashboard.registry.json');

interface FixtureCase {
  utterance: string;
  intent: string;
  intentConf: number;
  slots: Array<{ name: string; value: string; start: number; end: number }>;
}
interface Fixture {
  artifactsDir: string;
  cases: FixtureCase[];
}

const fixture: Fixture | null = fs.existsSync(FIXTURE) ? (JSON.parse(fs.readFileSync(FIXTURE, 'utf8')) as Fixture) : null;
const MODEL_DIR = fixture ? path.join(PKG, fixture.artifactsDir) : '';
const HAVE_MODEL = !!fixture && fs.existsSync(path.join(MODEL_DIR, 'ledger.json'));

describe.skipIf(!HAVE_MODEL)('Node ↔ Python session parity on the committed example model', () => {
  it('has a non-trivial fixture (non-vacuity)', () => {
    expect(fixture!.cases.length).toBeGreaterThanOrEqual(15);
    expect(fixture!.cases.some((c) => c.slots.length > 0)).toBe(true);
    expect(fixture!.cases.some((c) => c.slots.length === 0)).toBe(true);
    expect(new Set(fixture!.cases.map((c) => c.intent)).size).toBeGreaterThanOrEqual(10);
  });

  it('reproduces intent, confidence (±0.01) and slot spans for every fixture case', async () => {
    const reg = loadRegistryFile(EXAMPLE);
    const art = await loadArtifacts(MODEL_DIR, reg);
    const mismatches: string[] = [];
    for (const c of fixture!.cases) {
      const r = await classify(c.utterance, art, reg);
      if (r.intent !== c.intent) mismatches.push(`intent ${JSON.stringify(c.utterance)}: node=${r.intent} py=${c.intent}`);
      if (Math.abs(r.intentConf - c.intentConf) > 0.01)
        mismatches.push(`conf ${JSON.stringify(c.utterance)}: node=${r.intentConf.toFixed(4)} py=${c.intentConf.toFixed(4)}`);
      const nodeSlots = r.slots.map((s) => ({ name: s.name, value: s.value, start: s.start, end: s.end }));
      if (JSON.stringify(nodeSlots) !== JSON.stringify(c.slots))
        mismatches.push(`slots ${JSON.stringify(c.utterance)}: node=${JSON.stringify(nodeSlots)} py=${JSON.stringify(c.slots)}`);
    }
    expect(mismatches, mismatches.join('\n')).toEqual([]);
  });
});
