import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createEngine, RegistryValidationError, type Registry } from '../core/index';

const reg: Registry = {
  registryVersion: '1',
  app: { slug: 'e', name: 'E' },
  slots: {},
  intents: [
    { id: 'open_reports', family: 'howto', label: 'Find the reports', keywords: 'reports', templates: ['x'], answer: { steps: ['Open Reports.'], links: [{ label: 'Reports', target: '/r', need: 'reports' }] } },
    { id: 'report_count', family: 'status', label: 'How many reports', keywords: 'count', templates: ['y'], status: { unavailable: 'Cannot read counts here.' } },
  ],
};

function sha(s: string | Buffer): string {
  return createHash('sha256').update(s).digest('hex');
}

function fakeBundle(opts: { corruptSha?: boolean }): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cortex-bundle-'));
  const onnx = Buffer.from('not really an onnx graph');
  const tok = JSON.stringify({ version: '1', lower: true, maxLen: 4, padId: 0, unkId: 1, padToken: '<pad>', unkToken: '<unk>', vocab: {} });
  const labels = JSON.stringify({ version: '1', intents: ['open_reports', 'report_count', 'greeting', 'out_of_domain'], slots: ['O'] });
  fs.writeFileSync(path.join(dir, 'm.onnx'), onnx);
  fs.writeFileSync(path.join(dir, 'm.tokenizer.json'), tok);
  fs.writeFileSync(path.join(dir, 'm.labels.json'), labels);
  fs.writeFileSync(
    path.join(dir, 'ledger.json'),
    JSON.stringify({
      version: '9.9.9',
      onnx: { file: 'm.onnx', sha256: opts.corruptSha ? 'deadbeef' : sha(onnx), bytes: onnx.length, quantized: false },
      tokenizer: { file: 'm.tokenizer.json', sha256: sha(tok) },
      labels: { file: 'm.labels.json', sha256: sha(labels) },
      trainedFrom: { registryHash: 'x', datasetHash: 'y', seed: 42 },
      metrics: { inDistribution: { intentAccuracy: 1, perIntentF1: {} }, heldout: null },
      acceptanceFloor: { inDistributionIntentAccuracy: 0.9, heldoutIntentAccuracy: 0.85 },
    }),
  );
  return dir;
}

describe('engine', () => {
  it('no artifactsDir ⇒ keyword-only, answers via keyword/fallback, modelVersion null', async () => {
    const e = createEngine({ registry: reg, isAllowed: () => false, status: () => 'four open' });
    expect(e.mode).toBe('keyword-only');
    expect(e.modelLoadError).toBe('no artifactsDir supplied');
    const a = await e.answer('where are the reports');
    expect(a).toMatchObject({ intent: 'open_reports', ladderRung: 'keyword', modelVersion: null, intentConf: null, answerText: '1. Open Reports.' });
    expect(a.chips).toEqual([]);
    expect(a.evidence).toEqual({
      modelVersion: null,
      modelSha256: null,
      intent: 'open_reports',
      intentConf: null,
      ladderRung: 'keyword',
      plannerTemplateId: 'planner.howto.open_reports',
      registryEntriesUsed: ['intent:open_reports'],
    });
    const s = await e.answer('what is the count');
    expect(s.answerText).toBe('four open');
    const f = await e.answer('sing me a song');
    expect(f).toMatchObject({ intent: null, ladderRung: 'fallback' });
    expect(f.evidence.plannerTemplateId).toBe('planner.fallback');
    await expect(e.classify('x')).rejects.toThrow(/keyword-only/);
  });

  it('missing artifactsDir path ⇒ keyword-only with modelLoadError, never a throw', async () => {
    const e = createEngine({ registry: reg, artifactsDir: '/definitely/not/here' });
    expect(e.mode).toBe('keyword-only');
    expect(e.modelLoadError).toMatch(/no ledger\.json/);
    expect((await e.answer('reports')).ladderRung).toBe('keyword');
  });

  it('sha256 mismatch ⇒ keyword-only with the mismatch recorded', async () => {
    const dir = fakeBundle({ corruptSha: true });
    try {
      const e = createEngine({ registry: reg, artifactsDir: dir });
      expect(e.mode).toBe('keyword-only');
      expect(e.modelLoadError).toMatch(/sha256 mismatch/);
      expect(e.modelLoadError).toMatch(/deadbeef/);
      expect(e.modelVersion).toBeNull();
      const a = await e.answer('reports');
      expect(a.ladderRung).toBe('keyword');
      expect(a.modelVersion).toBeNull();
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('valid ledger but unloadable graph ⇒ starts in model mode, degrades on warm() with the error recorded', async () => {
    const dir = fakeBundle({});
    try {
      const e = createEngine({ registry: reg, artifactsDir: dir });
      expect(e.mode).toBe('model');
      expect(e.modelVersion).toBe('9.9.9');
      await e.warm();
      expect(e.mode).toBe('keyword-only');
      expect(e.modelLoadError).toMatch(/model load failed/);
      const a = await e.answer('reports');
      expect(a.ladderRung).toBe('keyword');
      expect(a.modelVersion).toBeNull();
      await expect(e.classify('reports')).rejects.toThrow(/keyword-only/);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('an invalid registry fails loudly at create time', () => {
    expect(() => createEngine({ registry: { ...reg, intents: [] } })).toThrow(RegistryValidationError);
  });
});
