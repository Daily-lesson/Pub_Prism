import { describe, expect, it } from 'vitest';
import { resolveLadder, buildKeywordBags, normalizeRegistry, type ClassifyResult, type Registry } from '../core/index';

const R = normalizeRegistry({
  registryVersion: '1',
  app: { slug: 'l', name: 'L' },
  slots: {},
  intents: [
    { id: 'open_reports', family: 'howto', label: 'Find the reports', keywords: 'reports', templates: ['x'], answer: { steps: ['s'] } },
    { id: 'export_data', family: 'howto', label: 'Export data', keywords: 'export csv', templates: ['y'], answer: { steps: ['s'] } },
  ],
} as Registry);
const bags = buildKeywordBags(R);

const stub =
  (intent: string, intentConf: number): ((q: string) => Promise<ClassifyResult>) =>
  async () => ({ intent, intentConf, slots: [], modelVersion: '0.1.0', modelSha256: 'abc' });

describe('ladder (§5.1)', () => {
  it('keyword-only mode works with no classifier', async () => {
    expect(await resolveLadder('where are the reports', { bags })).toEqual({ rung: 'keyword', intent: 'open_reports', intentConf: null, slots: [], modelIntent: null });
    expect(await resolveLadder('what is the weather', { bags })).toEqual({ rung: 'fallback', intent: null, intentConf: null, slots: [], modelIntent: null });
  });

  it('model ≥ threshold ⇒ cortex rung (including out_of_domain)', async () => {
    const r = await resolveLadder('anything', { bags, classify: stub('export_data', 0.9), threshold: 0.6 });
    expect(r.rung).toBe('cortex');
    expect(r.intent).toBe('export_data');
    expect(r.intentConf).toBe(0.9);
    const ood = await resolveLadder('reports please', { bags, classify: stub('out_of_domain', 0.95) });
    expect(ood).toMatchObject({ rung: 'cortex', intent: 'out_of_domain' });
  });

  it('below threshold ⇒ keyword rung keeps the model conf; no keyword hit ⇒ fallback', async () => {
    const r = await resolveLadder('export it as csv', { bags, classify: stub('open_reports', 0.4) });
    expect(r).toMatchObject({ rung: 'keyword', intent: 'export_data', intentConf: 0.4, modelIntent: 'open_reports' });
    const f = await resolveLadder('sing a song', { bags, classify: stub('open_reports', 0.59) });
    expect(f).toMatchObject({ rung: 'fallback', intent: null, intentConf: 0.59 });
    const exactly = await resolveLadder('x', { bags, classify: stub('open_reports', 0.6) });
    expect(exactly.rung).toBe('cortex');
  });

  it('a throwing classifier degrades to keyword for that query and records the error', async () => {
    const r = await resolveLadder('reports', {
      bags,
      classify: async () => {
        throw new Error('session gone');
      },
    });
    expect(r).toMatchObject({ rung: 'keyword', intent: 'open_reports', intentConf: null, classifyError: 'session gone' });
  });
});
