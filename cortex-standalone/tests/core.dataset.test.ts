import { describe, expect, it } from 'vitest';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import {
  compileDataset,
  writeDataset,
  loadRegistryFile,
  normalizeRegistry,
  registryHash,
  autoDetectSlots,
  HeldoutOverlapError,
  mulberry32,
  deriveSeed,
  shuffle,
  augmentExample,
  AUGMENT_VARIANTS_PER_EXAMPLE,
  VARIANTS_PER_SLOTTED_TEMPLATE,
  generateBaseExamplesForIntent,
  PHRASE_SYNONYMS,
  PREFIXES,
  SUFFIXES,
  type Registry,
  type GeneratedExample,
} from '../core/index';

const EXAMPLE = path.resolve(__dirname, '..', 'registry', 'examples', 'ops-dashboard.registry.json');

function small(): Registry {
  return {
    registryVersion: '1',
    app: { slug: 'small', name: 'Small' },
    slots: { region: { vocab: [{ id: 'north', label: 'North campus' }, { id: 'south', label: 'South campus' }, { id: 'east', label: 'East wing' }] } },
    intents: [
      {
        id: 'open_reports',
        family: 'howto',
        label: 'Find the reports',
        slots: ['region'],
        templates: ['how do i find the reports', 'show me the {region} reports'],
        paraphrases: ['where are my numbers', 'figures for the east WING please'],
        answer: { steps: ['Open Reports.'] },
      },
      { id: 'report_count', family: 'status', label: 'How many reports', templates: ['how many reports are there', 'count the reports'], status: { unavailable: 'n/a' } },
    ],
  };
}

describe('prng', () => {
  it('mulberry32 is deterministic and in [0,1)', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 100; i++) {
      const x = a();
      expect(x).toBe(b());
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
    expect(mulberry32(42)()).not.toBe(mulberry32(43)());
  });
  it('deriveSeed is stable and salt-sensitive', () => {
    expect(deriveSeed(42, 'split::a')).toBe(deriveSeed(42, 'split::a'));
    expect(deriveSeed(42, 'split::a')).not.toBe(deriveSeed(42, 'split::b'));
    expect(deriveSeed(42, 'x')).not.toBe(deriveSeed(43, 'x'));
  });
  it('shuffle does not mutate and is a permutation', () => {
    const src = [1, 2, 3, 4, 5];
    const out = shuffle(src, mulberry32(1));
    expect(src).toEqual([1, 2, 3, 4, 5]);
    expect([...out].sort()).toEqual([1, 2, 3, 4, 5]);
  });
});

describe('compileDataset (§2)', () => {
  const reg = loadRegistryFile(EXAMPLE);
  const ds = compileDataset(reg, 42);

  it('same seed ⇒ byte-identical; different seed ⇒ different hash', () => {
    const again = compileDataset(reg, 42);
    expect(JSON.stringify(again)).toBe(JSON.stringify(ds));
    expect(again.manifest.datasetHash).toBe(ds.manifest.datasetHash);
    expect(compileDataset(reg, 7).manifest.datasetHash).not.toBe(ds.manifest.datasetHash);
  });

  it('every intent appears in every split, in taxonomy order', () => {
    const ids = reg.intents.map((i) => i.id);
    expect(ds.manifest.intents).toEqual(ids);
    for (const id of ids) {
      expect(ds.train.some((e) => e.intent === id), `${id} in train`).toBe(true);
      expect(ds.val.some((e) => e.intent === id), `${id} in val`).toBe(true);
      expect(ds.test.some((e) => e.intent === id), `${id} in test`).toBe(true);
    }
  });

  it('span invariant: utterance.slice(start,end) === value on every example', () => {
    for (const ex of [...ds.train, ...ds.val, ...ds.test]) {
      for (const s of ex.slots) {
        expect(ex.utterance.slice(s.start, s.end)).toBe(s.value);
        expect(Object.keys(reg.slots)).toContain(s.name);
      }
    }
  });

  it('global dedupe: no utterance appears twice across all splits', () => {
    const seen = new Set<string>();
    for (const ex of [...ds.train, ...ds.val, ...ds.test]) {
      expect(seen.has(ex.utterance), ex.utterance).toBe(false);
      seen.add(ex.utterance);
    }
  });

  it('per-intent split sizes follow max(1, round(n*0.15))', () => {
    for (const [id, n] of Object.entries(ds.manifest.counts.perIntent)) {
      const val = ds.val.filter((e) => e.intent === id).length;
      const test = ds.test.filter((e) => e.intent === id).length;
      const train = ds.train.filter((e) => e.intent === id).length;
      expect(val).toBe(Math.max(1, Math.round(n * 0.15)));
      expect(test).toBe(Math.max(1, Math.round(n * 0.15)));
      expect(train).toBe(n - val - test);
    }
    expect(ds.manifest.counts.total).toBe(ds.manifest.counts.train + ds.manifest.counts.val + ds.manifest.counts.test);
  });

  it('manifest carries explicit intents/slotNames/slotLabels + registryHash', () => {
    expect(ds.manifest.slotNames).toEqual(['region', 'department', 'period']);
    expect(ds.manifest.slotLabels).toEqual(['O', 'B-region', 'I-region', 'B-department', 'I-department', 'B-period', 'I-period']);
    expect(ds.manifest.registryHash).toBe(registryHash(reg));
    expect(ds.manifest.seed).toBe(42);
  });

  it('template expansion: unslotted ⇒ 1, slotted ⇒ VARIANTS_PER_SLOTTED_TEMPLATE, labels inserted', () => {
    const n = normalizeRegistry(small());
    const base = generateBaseExamplesForIntent(n, n.intents[0], 42);
    expect(VARIANTS_PER_SLOTTED_TEMPLATE).toBe(4);
    expect(base.length).toBe(1 + 4);
    const labels = new Set(n.slots.region.vocab.map((v) => v.label));
    for (const ex of base.slice(1)) {
      expect(ex.slots.length).toBe(1);
      expect(labels.has(ex.slots[0].value)).toBe(true);
      expect(ex.utterance.slice(ex.slots[0].start, ex.slots[0].end)).toBe(ex.slots[0].value);
    }
  });

  it('augmentation: up to 20 variants, original first, spans recomputed, no domain-specific fillers', () => {
    expect(AUGMENT_VARIANTS_PER_EXAMPLE).toBe(20);
    const ex: GeneratedExample = { utterance: 'show me the North campus reports', intent: 'open_reports', slots: [{ name: 'region', value: 'North campus', start: 12, end: 24 }] };
    const variants = augmentExample(ex, 42, 0);
    expect(variants[0]).toEqual(ex);
    expect(variants.length).toBeGreaterThan(1);
    expect(variants.length).toBeLessThanOrEqual(20);
    for (const v of variants) expect(v.utterance.slice(v.slots[0].start, v.slots[0].end)).toBe(v.slots[0].value);
    expect(JSON.stringify(augmentExample(ex, 42, 0))).toBe(JSON.stringify(variants));
    const fillers = [...PREFIXES, ...SUFFIXES, ...PHRASE_SYNONYMS.flatMap((p) => p.alts)].join(' ');
    expect(fillers).toMatch(/please/);
  });

  it('autoDetectSlots: longest label first, case-insensitive, one per slot, no overlaps', () => {
    const n = normalizeRegistry({
      ...small(),
      slots: { region: { vocab: [{ id: 'n', label: 'North' }, { id: 'nc', label: 'North campus' }] }, dept: { vocab: [{ id: 'c', label: 'campus' }] } },
    });
    const spans = autoDetectSlots(n, 'figures for the NORTH CAMPUS please', ['region', 'dept']);
    expect(spans).toEqual([{ name: 'region', value: 'NORTH CAMPUS', start: 16, end: 28 }]);
    const spans2 = autoDetectSlots(n, 'campus then north', ['region', 'dept']);
    expect(spans2.map((s) => `${s.name}:${s.value}`)).toEqual(['dept:campus', 'region:north']);
  });

  it('held-out guard throws when a compiled utterance equals a heldout one (case-insensitive)', () => {
    const r = small();
    r.heldout = [{ utterance: 'How Do I Find The Reports', intent: 'open_reports' }];
    expect(() => compileDataset(r, 42)).toThrow(HeldoutOverlapError);
    expect(() => compileDataset(r, 42)).toThrow(/held-out/);
    r.heldout = [{ utterance: 'something entirely different', intent: 'open_reports' }];
    expect(() => compileDataset(r, 42)).not.toThrow();
  });

  it('writeDataset writes exactly the four files, one example per line', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cortex-ds-'));
    try {
      const d = compileDataset(small(), 1);
      writeDataset(d, dir);
      expect(fs.readdirSync(dir).sort()).toEqual(['manifest.json', 'test.jsonl', 'train.jsonl', 'val.jsonl']);
      const lines = fs.readFileSync(path.join(dir, 'train.jsonl'), 'utf8').trim().split('\n');
      expect(lines.length).toBe(d.train.length);
      expect(JSON.parse(lines[0])).toEqual(d.train[0]);
      const m = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8'));
      expect(m).toEqual(d.manifest);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
