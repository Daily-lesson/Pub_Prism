/**
 * The dataset compiler (docs/CONTRACTS.md §2). Deterministic: same registry + same seed ⇒
 * byte-identical output.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { canonicalJson, sha256Hex, registryHash } from '../registry/hash';
import { normalizeRegistry } from '../registry/validate';
import { slotLabelsOf, slotNamesOf, type NormalizedRegistry, type Registry } from '../registry/types';
import { generateFromTemplates } from './grammar';
import { paraphraseExamples } from './paraphrases';
import { assertNoHeldoutOverlap, heldoutUtteranceSet } from './heldoutGuard';
import { deriveSeed, mulberry32, shuffle } from './prng';
import type { GeneratedExample } from './types';

export const TRAIN_RATIO = 0.7;
export const VAL_RATIO = 0.15;
export const TEST_RATIO = 0.15;

export interface DatasetManifest {
  /** The registry's app identity, so the trainer needs no separate --slug (§2.8). */
  app: { slug: string; name: string };
  seed: number;
  registryHash: string;
  datasetHash: string;
  intents: string[];
  slotNames: string[];
  slotLabels: string[];
  counts: { train: number; val: number; test: number; total: number; perIntent: Record<string, number> };
}

export interface CompiledDataset {
  train: GeneratedExample[];
  val: GeneratedExample[];
  test: GeneratedExample[];
  manifest: DatasetManifest;
}

/** Per-intent split sizes (§2.5): val/test = max(1, round(n·0.15)), train = the rest. */
export function splitCounts(n: number): { train: number; val: number; test: number } {
  if (n <= 0) return { train: 0, val: 0, test: 0 };
  if (n < 3) return { train: n, val: 0, test: 0 };
  let val = Math.max(1, Math.round(n * VAL_RATIO));
  let test = Math.max(1, Math.round(n * TEST_RATIO));
  let train = n - val - test;
  if (train < 1) {
    train = 1;
    const remaining = n - train;
    val = Math.max(1, Math.floor(remaining / 2));
    test = remaining - val;
    if (test < 1) {
      test = 1;
      val = Math.max(0, remaining - test);
    }
  }
  return { train, val, test };
}

export function compileDataset(registryInput: Registry | NormalizedRegistry, seed: number): CompiledDataset {
  if (!Number.isFinite(seed)) throw new Error('compileDataset: seed must be a finite number');
  const registry = normalizeRegistry(registryInput);

  const train: GeneratedExample[] = [];
  const val: GeneratedExample[] = [];
  const test: GeneratedExample[] = [];
  const perIntent: Record<string, number> = {};

  // Global cross-intent dedupe on the exact utterance string (§2.4) — taxonomy order is fixed,
  // so an earlier-declared intent wins deterministically.
  const globalSeen = new Set<string>();

  for (const intent of registry.intents) {
    const rawPool = [...generateFromTemplates(registry, intent, seed), ...paraphraseExamples(registry, intent)];
    const pool = rawPool.filter((ex) => {
      if (globalSeen.has(ex.utterance)) return false;
      globalSeen.add(ex.utterance);
      return true;
    });
    const rng = mulberry32(deriveSeed(seed, `split::${intent.id}`));
    const shuffled = shuffle(pool, rng);
    const { train: nTrain, val: nVal, test: nTest } = splitCounts(shuffled.length);
    if (nTrain < 1 || nVal < 1 || nTest < 1) {
      throw new Error(
        `intent "${intent.id}" compiles to only ${shuffled.length} example(s) after dedupe — too few to appear in ` +
          `every split (rule 2.5). Add templates or paraphrases.`,
      );
    }

    train.push(...shuffled.slice(0, nTrain));
    val.push(...shuffled.slice(nTrain, nTrain + nVal));
    test.push(...shuffled.slice(nTrain + nVal, nTrain + nVal + nTest));
    perIntent[intent.id] = shuffled.length;
  }

  const trainOut = shuffle(train, mulberry32(deriveSeed(seed, 'global::train')));
  const valOut = shuffle(val, mulberry32(deriveSeed(seed, 'global::val')));
  const testOut = shuffle(test, mulberry32(deriveSeed(seed, 'global::test')));

  assertNoHeldoutOverlap([...trainOut, ...valOut, ...testOut].map((ex) => ex.utterance), heldoutUtteranceSet(registry.heldout));

  const datasetHash = sha256Hex(canonicalJson({ seed, train: trainOut, val: valOut, test: testOut }));

  return {
    train: trainOut,
    val: valOut,
    test: testOut,
    manifest: {
      app: { slug: registry.app.slug, name: registry.app.name },
      seed,
      registryHash: registryHash(registry),
      datasetHash,
      intents: registry.intents.map((i) => i.id),
      slotNames: slotNamesOf(registry),
      slotLabels: slotLabelsOf(registry),
      counts: {
        train: trainOut.length,
        val: valOut.length,
        test: testOut.length,
        total: trainOut.length + valOut.length + testOut.length,
        perIntent,
      },
    },
  };
}

function toJsonl(examples: readonly GeneratedExample[]): string {
  return examples.map((ex) => JSON.stringify(ex)).join('\n') + (examples.length ? '\n' : '');
}

/** Write exactly `train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json` (§2.7–§2.8). */
export function writeDataset(dataset: CompiledDataset, outDir: string): void {
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, 'train.jsonl'), toJsonl(dataset.train), 'utf8');
  writeFileSync(join(outDir, 'val.jsonl'), toJsonl(dataset.val), 'utf8');
  writeFileSync(join(outDir, 'test.jsonl'), toJsonl(dataset.test), 'utf8');
  writeFileSync(join(outDir, 'manifest.json'), JSON.stringify(dataset.manifest, null, 2) + '\n', 'utf8');
}
